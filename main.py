"""
Crawl4AI Adaptive Crawler with LOCAL MULTILINGUAL EMBEDDING Strategy + OpenRouter Re-ranking + DeepSeek Reasoner

VERSION 3.7.1 - SPA/JavaScript FIX + EXTRACTION FIXES

ROOT CAUSE FIXES (identified via Context7 documentation):
1. SPAFriendlyCrawler wrapper - AdaptiveCrawler doesn't accept CrawlerRunConfig,
   so SPA/JavaScript sites weren't rendering. The wrapper intercepts arun() calls
   and automatically injects CrawlerRunConfig with wait_for, process_iframes, etc.
2. Embedding model verification at startup - detects model loading failures early
3. Verbose logging for debugging - helps trace content extraction issues
4. FIXED: CrawlResult.markdown extraction - was looking for .content instead of .markdown.raw_markdown
5. FIXED: Format string bugs - .0% formatter with manual multiplication
6. FIXED: Early validation of extracted content - fail fast if KB extraction fails

EMBEDDING STRATEGY (Semantic Crawling - NO API NEEDED!):
- Uses LOCAL sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- Supports 50+ languages including Chinese, English, and more
- AGGRESSIVE EXPLORATION: confidence_threshold=0.05, min_relative_improvement=0.01
- Disables early stopping to ensure thorough crawling

SPA/JavaScript Support (NEW in v3.7.0):
- wait_for="css:body" ensures JavaScript content renders before extraction
- process_iframes=True includes iframe content
- delay_before_return_html=2.0s extra delay for JS execution
- page_timeout=60000ms longer timeout for slow sites

FALLBACK (if use_embeddings=false):
- Uses BM25 statistical strategy (keyword-based)

RE-RANKING (optional, requires OPENROUTER_API_KEY):
- OpenRouter embeddings (qwen3-embedding-8b) for semantic re-ranking

ANSWER GENERATION (requires DEEPSEEK_API_KEY):
- DeepSeek-reasoner for comprehensive answers

Version: 3.7.1 (SPA FIX + EXTRACTION FIXES - Context7 Verified)
"""

import os
import sys
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional, List
from urllib.parse import urlparse
import re

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Core crawl4ai imports
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from crawl4ai.async_configs import CacheMode
    print("Core crawl4ai imported successfully", flush=True)
except ImportError as e:
    print(f"Failed to import core crawl4ai: {e}", flush=True)
    sys.exit(1)


# ============================================================================
# FIX: AsyncWebCrawler Wrapper for SPA/JavaScript Sites
# ============================================================================
# ROOT CAUSE: AdaptiveCrawler doesn't accept CrawlerRunConfig, so it can't
# pass wait_for, process_iframes, or wait_until to handle SPA sites.
# SOLUTION: Wrap AsyncWebCrawler to automatically inject CrawlerRunConfig
# with SPA-friendly settings on every arun() call.
# ============================================================================

class SPAFriendlyCrawler:
    """
    Wrapper around AsyncWebCrawler that automatically adds SPA-friendly settings.

    This solves the issue where AdaptiveCrawler can't pass CrawlerRunConfig
    because it calls crawler.arun() internally without config parameter.

    Settings applied:
    - wait_for: Wait for body content to render
    - process_iframes: Include iframe content
    - delay_before_return_html: Extra delay for JS to execute
    - page_timeout: Longer timeout for slow sites
    """

    def __init__(self, crawler: AsyncWebCrawler, default_config: CrawlerRunConfig = None):
        self._crawler = crawler
        self._default_config = default_config or CrawlerRunConfig(
            # Wait for JavaScript content to render
            wait_for="css:body",
            # Include iframe content (important for embedded content)
            process_iframes=True,
            # Extra delay for JavaScript execution
            delay_before_return_html=2.0,
            # Longer timeout for slow sites
            page_timeout=60000,
            # Bypass cache to get fresh content
            cache_mode=CacheMode.BYPASS,
            # Be verbose for debugging
            verbose=True
        )
        print(f"SPAFriendlyCrawler initialized with SPA-friendly defaults", flush=True)
        print(f"  - wait_for: {self._default_config.wait_for}", flush=True)
        print(f"  - process_iframes: {self._default_config.process_iframes}", flush=True)
        print(f"  - delay_before_return_html: {self._default_config.delay_before_return_html}s", flush=True)

    async def arun(self, url: str, config: CrawlerRunConfig = None, **kwargs):
        """
        Intercept arun() calls and inject SPA-friendly config.

        If AdaptiveCrawler calls arun() without config, we use our default.
        If a config is provided, we could merge settings (for now we use provided).
        """
        # Use provided config or fall back to our SPA-friendly defaults
        effective_config = config if config is not None else self._default_config

        print(f"SPAFriendlyCrawler.arun() called for: {url[:80]}...", flush=True)
        print(f"  Using config with wait_for={effective_config.wait_for}", flush=True)

        return await self._crawler.arun(url=url, config=effective_config, **kwargs)

    # Delegate all other methods/attributes to the underlying crawler
    def __getattr__(self, name):
        return getattr(self._crawler, name)

    async def __aenter__(self):
        await self._crawler.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._crawler.__aexit__(exc_type, exc_val, exc_tb)


# ============================================================================
# FIX: Embedding Model Verification at Startup
# ============================================================================
# ROOT CAUSE: The embedding model might fail silently, causing 0 terms extraction.
# SOLUTION: Verify the model loads and generates embeddings at startup.
# ============================================================================

EMBEDDING_MODEL_VERIFIED = False
EMBEDDING_MODEL_ERROR = None

def verify_embedding_model(model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> bool:
    """
    Verify that the embedding model can be loaded and generates valid embeddings.

    This runs at startup to catch model loading issues early.
    """
    global EMBEDDING_MODEL_VERIFIED, EMBEDDING_MODEL_ERROR

    try:
        print(f"Verifying embedding model: {model_name}...", flush=True)

        # Try to import sentence-transformers
        from sentence_transformers import SentenceTransformer

        # Load the model
        model = SentenceTransformer(model_name)
        print(f"  ✓ Model loaded successfully", flush=True)

        # Test with sample texts (include Chinese for multilingual verification)
        test_texts = [
            "This is a test sentence in English.",
            "这是一个中文测试句子。",  # Chinese test
            "澳門法律 刑事 販毒"  # Macau law terms
        ]

        # Generate embeddings
        embeddings = model.encode(test_texts)
        print(f"  ✓ Generated {len(embeddings)} embeddings", flush=True)
        print(f"  ✓ Embedding dimension: {len(embeddings[0])}", flush=True)

        # Verify embeddings are not zero vectors
        import numpy as np
        for i, (text, emb) in enumerate(zip(test_texts, embeddings)):
            norm = np.linalg.norm(emb)
            if norm < 0.001:
                raise ValueError(f"Embedding for '{text[:30]}...' has near-zero norm: {norm}")
            print(f"  ✓ Text {i+1} embedding norm: {norm:.4f}", flush=True)

        # Test similarity between Chinese texts
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity([embeddings[1]], [embeddings[2]])[0][0]
        print(f"  ✓ Chinese text similarity test: {sim:.4f}", flush=True)

        EMBEDDING_MODEL_VERIFIED = True
        print(f"✅ Embedding model verification PASSED", flush=True)
        return True

    except ImportError as e:
        EMBEDDING_MODEL_ERROR = f"sentence-transformers not installed: {e}"
        print(f"❌ Embedding model verification FAILED: {EMBEDDING_MODEL_ERROR}", flush=True)
        return False
    except Exception as e:
        EMBEDDING_MODEL_ERROR = str(e)
        print(f"❌ Embedding model verification FAILED: {EMBEDDING_MODEL_ERROR}", flush=True)
        import traceback
        traceback.print_exc()
        return False


# Adaptive crawler imports
try:
    from crawl4ai import AdaptiveCrawler, AdaptiveConfig
    ADAPTIVE_AVAILABLE = True
    print("AdaptiveCrawler imported successfully", flush=True)
except ImportError as e:
    print(f"AdaptiveCrawler not available: {e}", flush=True)
    ADAPTIVE_AVAILABLE = False

# Fallback: Deep crawling imports (if adaptive not available)
try:
    from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
    from crawl4ai.deep_crawling.filters import FilterChain, DomainFilter
    from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
    DEEP_CRAWL_AVAILABLE = True
    print("Deep crawling (fallback) imported successfully", flush=True)
except ImportError as e:
    print(f"Deep crawling not available: {e}", flush=True)
    DEEP_CRAWL_AVAILABLE = False


# ============================================================================
# Custom OpenRouter Embedding Client with Retry Logic
# ============================================================================

class OpenRouterEmbeddings:
    """Custom client for OpenRouter embeddings API with retry logic."""

    def __init__(self, api_key: str, model: str = "qwen/qwen3-embedding-8b", max_retries: int = 3):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/embeddings"
        self.max_retries = max_retries

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts from OpenRouter with retry logic."""
        if not texts:
            return []

        total_wait = 0
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self.base_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model,
                            "input": texts
                        }
                    )

                    if response.status_code == 429:  # Rate limited
                        wait_time = 2 ** attempt
                        total_wait += wait_time
                        print(f"Rate limited. Waiting {wait_time}s before retry (total wait: {total_wait}s)...", flush=True)
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status_code != 200:
                        print(f"OpenRouter embedding error: {response.status_code} - {response.text}", flush=True)
                        if attempt == self.max_retries - 1:
                            raise HTTPException(
                                status_code=response.status_code,
                                detail=f"OpenRouter embedding error: {response.text}"
                            )
                        continue

                    result = response.json()
                    # Extract embeddings in order
                    embeddings = [None] * len(texts)
                    for item in result.get("data", []):
                        idx = item.get("index", 0)
                        embeddings[idx] = item.get("embedding", [])

                    return embeddings

            except httpx.TimeoutException:
                if attempt == self.max_retries - 1:
                    print(f"Timeout after {self.max_retries} attempts", flush=True)
                    raise
                wait_time = 2 ** attempt
                total_wait += wait_time
                print(f"Timeout. Waiting {wait_time}s before retry (total wait: {total_wait}s)...", flush=True)
                await asyncio.sleep(wait_time)
                continue

        return []

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0

    try:
        a = np.array(vec1)
        b = np.array(vec2)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}", flush=True)
        return 0.0


async def rerank_by_embeddings(
    query: str,
    pages: List[dict],
    embedding_client: OpenRouterEmbeddings,
    top_k: int = 10
) -> List[dict]:
    """Re-rank pages by semantic similarity to query using embeddings.
    
    SOLUTION 3: Better weighting - trust semantic embeddings more (75% vs 25% BM25)
    """

    if not pages:
        return []

    print(f"Re-ranking {len(pages)} pages using OpenRouter embeddings...", flush=True)

    # Get query embedding
    query_embedding = await embedding_client.get_embedding(query)

    if not query_embedding:
        print("Failed to get query embedding, returning original order", flush=True)
        return pages[:top_k]

    # Get embeddings for page content (use first 4000 chars for better coverage)
    page_texts = []
    for page in pages:
        content = page.get('content', '')
        # Use more content for better semantic matching (especially for legal docs)
        text = content[:4000] if content else ''
        page_texts.append(text)

    # Batch embed all pages
    try:
        page_embeddings = await embedding_client.get_embeddings(page_texts)
    except Exception as e:
        print(f"Failed to get page embeddings: {e}, returning original order", flush=True)
        return pages[:top_k]

    # Calculate similarity scores
    scored_pages = []
    for i, page in enumerate(pages):
        if i < len(page_embeddings) and page_embeddings[i]:
            similarity = cosine_similarity(query_embedding, page_embeddings[i])
        else:
            similarity = 0.0

        # SOLUTION 3: Better weighting - trust semantic similarity more
        original_score = page.get('score', 0.5)
        # Weighted combination: 75% embedding, 25% BM25 (instead of 60/40)
        combined_score = (0.75 * similarity) + (0.25 * original_score)

        scored_pages.append({
            **page,
            'embedding_score': round(similarity, 4),
            'original_score': original_score,
            'score': round(combined_score, 4)
        })

    # Sort by combined score
    scored_pages.sort(key=lambda x: x['score'], reverse=True)

    print(f"Re-ranking complete. Top scores: {[p['score'] for p in scored_pages[:5]]}", flush=True)

    return scored_pages[:top_k]


# ============================================================================
# Helper: Language Detection
# ============================================================================

def detect_query_languages(query: str) -> dict:
    """Detect languages in query (Chinese, English, etc.)."""
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
    has_english = bool(re.search(r'[a-zA-Z]', query))
    
    languages = []
    if has_chinese:
        languages.append("Chinese")
    if has_english:
        languages.append("English")
    
    return {
        "has_chinese": has_chinese,
        "has_english": has_english,
        "languages": languages,
        "is_multilingual": len(languages) > 1
    }


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler."""
    openrouter_configured = bool(os.environ.get("OPENROUTER_API_KEY"))
    print("=" * 60, flush=True)
    print("Crawl4AI Adaptive Crawler v3.7.1 (SPA + EXTRACTION FIXES) Starting...", flush=True)
    print(f"AdaptiveCrawler Available: {ADAPTIVE_AVAILABLE}", flush=True)
    print(f"Deep Crawl Fallback Available: {DEEP_CRAWL_AVAILABLE}", flush=True)
    print(f"OpenRouter API Key: {'Configured' if openrouter_configured else 'NOT SET'}", flush=True)
    print("=" * 60, flush=True)

    # FIX: Verify embedding model at startup
    print("STARTUP VERIFICATION:", flush=True)
    embedding_ok = verify_embedding_model()
    if not embedding_ok:
        print(f"⚠️  WARNING: Embedding model verification failed!", flush=True)
        print(f"   Error: {EMBEDDING_MODEL_ERROR}", flush=True)
        print(f"   The embedding strategy may not work correctly.", flush=True)
    print("=" * 60, flush=True)

    print("FIXES APPLIED (v3.7.1):", flush=True)
    print("  ✓ FIX 1: SPAFriendlyCrawler wrapper for JavaScript sites", flush=True)
    print("    - Automatically injects CrawlerRunConfig to AdaptiveCrawler", flush=True)
    print("    - wait_for='css:body' ensures content renders", flush=True)
    print("    - process_iframes=True includes embedded content", flush=True)
    print("    - delay_before_return_html=2.0s for JS execution", flush=True)
    print("  ✓ FIX 2: Embedding model verification at startup", flush=True)
    print("  ✓ FIX 3: Verbose logging for debugging", flush=True)
    print("  ✓ FIX 4: CrawlResult.markdown extraction (was looking for .content)", flush=True)
    print("  ✓ FIX 5: Format string bugs (:.0% with manual multiplication)", flush=True)
    print("  ✓ FIX 6: Early validation of extracted content (fail fast)", flush=True)
    print("=" * 60, flush=True)

    print("EMBEDDING Strategy (LOCAL MULTILINGUAL - No API needed!):", flush=True)
    print("  ✓ Model: paraphrase-multilingual-mpnet-base-v2 (50+ languages)", flush=True)
    print(f"  ✓ Model Verified: {'YES' if EMBEDDING_MODEL_VERIFIED else 'NO'}", flush=True)
    print("  ✓ AGGRESSIVE exploration (confidence=0.05, min_improvement=0.01)", flush=True)
    print("  ✓ Query variations=20 for comprehensive coverage", flush=True)
    print("  ✓ max_pages=50, top_k_links=25 (deeper crawling)", flush=True)
    if openrouter_configured:
        print("  ✓ Re-ranking 75% semantic + 25% BM25", flush=True)
        print("  ✓ Re-ranking: OpenRouter qwen3-embedding-8b with retry logic", flush=True)
    print("  ✓ Answer Generation: DeepSeek-reasoner", flush=True)
    print("=" * 60, flush=True)
    yield
    print("Shutting down...", flush=True)


app = FastAPI(
    title="Crawl4AI Adaptive Crawler",
    description="Intelligent web crawler with LOCAL MULTILINGUAL EMBEDDING strategy (50+ languages) + SPA/JavaScript support + OpenRouter re-ranking + DeepSeek reasoning - v3.7.1 with extraction fixes",
    version="3.7.1",
    lifespan=lifespan
)


class CrawlRequest(BaseModel):
    """Request model for adaptive crawling."""
    start_url: str
    query: str
    max_pages: Optional[int] = 50
    confidence_threshold: Optional[float] = 0.05
    use_embeddings: Optional[bool] = True
    embedding_model: Optional[str] = "qwen/qwen3-embedding-8b"


class CrawlResponse(BaseModel):
    """Response model with direct answer."""
    success: bool
    answer: str
    confidence: float
    pages_crawled: int
    sources: List[dict]
    message: str
    embedding_used: bool


async def call_deepseek(
    query: str,
    context: str,
    api_key: str,
    max_tokens: int = 3000,
    max_retries: int = 3
) -> str:
    """Call DeepSeek API to generate an answer with retry logic."""

    system_prompt = """You are a helpful assistant that provides direct, accurate answers based on the provided web content.

Your task:
1. Carefully analyze ALL the crawled web content provided
2. Find the most relevant and detailed information to answer the user's query
3. Provide a comprehensive, accurate answer in plain text
4. Include specific details, code examples, or steps if available in the content
5. If the information is incomplete, mention what was found and suggest where to look
6. Cite the source URLs for the information you use

Be thorough and informative. Extract maximum value from the crawled content."""

    user_message = f"""Query: {query}

Crawled Web Content:
{context}

Based on the above content, provide a detailed and accurate answer to the query."""

    total_wait = 0
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-reasoner",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.2
                    }
                )

                if response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    total_wait += wait_time
                    print(f"DeepSeek rate limited. Waiting {wait_time}s (total: {total_wait}s)...", flush=True)
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    if attempt == max_retries - 1:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"DeepSeek API error: {response.text}"
                        )
                    continue

                result = response.json()
                return result["choices"][0]["message"]["content"]

        except httpx.TimeoutException:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=504, detail="DeepSeek API timeout")
            wait_time = 2 ** attempt
            total_wait += wait_time
            print(f"DeepSeek timeout. Retrying in {wait_time}s (total wait: {total_wait}s)...", flush=True)
            await asyncio.sleep(wait_time)

    raise HTTPException(status_code=500, detail="Failed to get response from DeepSeek after retries")


def format_adaptive_context(relevant_pages: List[dict], max_chars: int = 25000) -> str:
    """Format relevant pages from AdaptiveCrawler into context for LLM."""
    context_parts = []
    total_chars = 0

    for i, page in enumerate(relevant_pages, 1):
        url = page.get('url', 'unknown')
        score = page.get('score', 0)
        embedding_score = page.get('embedding_score', None)
        content = page.get('content', '')

        if not content or len(content) < 50:
            continue

        # Truncate individual page content
        page_text = content[:5000] if len(content) > 5000 else content

        # ✅ FIXED: Proper format string without double multiplication
        if embedding_score is not None:
            score_info = f"Combined: {int(score * 100)}%, Semantic: {int(embedding_score * 100)}%"
        else:
            score_info = f"Relevance: {int(score * 100)}%"

        entry = f"""
=== Page {i} ({score_info}): {url} ===
{page_text}

"""
        if total_chars + len(entry) > max_chars:
            break

        context_parts.append(entry)
        total_chars += len(entry)

    return "\n".join(context_parts)


def extract_pages_from_result(result) -> List[dict]:
    """
    FIX: Extract pages from adaptive crawl result's knowledge base.
    
    CrawlState structure:
    - result.knowledge_base: List of CrawlResult objects
    - Each CrawlResult has:
      - .url: str
      - .markdown: MarkdownGenerationResult object
        - .raw_markdown: str (the actual content!)
        - .fit_markdown: str (filtered content, if available)
        - .fit_html: str (filtered HTML, if available)
      - .extracted_content: str (optional fallback)
    """
    pages = []
    
    print(f"\n>>> Extracting pages from result", flush=True)
    
    try:
        # Check if knowledge_base exists
        if not hasattr(result, 'knowledge_base') or not result.knowledge_base:
            print(f"  ⚠️  No knowledge_base found in result", flush=True)
            return []
        
        kb = result.knowledge_base
        kb_len = len(kb) if hasattr(kb, '__len__') else 'unknown'
        print(f"  ✓ Found {kb_len} items in knowledge base", flush=True)
        
        # Iterate through knowledge base items (should be CrawlResult objects)
        for i, doc in enumerate(kb):
            try:
                # Get URL
                url = None
                if hasattr(doc, 'url'):
                    url = doc.url
                else:
                    url = f'doc_{i}'
                
                # Extract content from .markdown object (MarkdownGenerationResult)
                content = None
                content_source = None
                
                if hasattr(doc, 'markdown') and doc.markdown:
                    md = doc.markdown
                    
                    # Try fit_markdown first (filtered content, more relevant)
                    if hasattr(md, 'fit_markdown') and md.fit_markdown:
                        content = md.fit_markdown
                        content_source = "fit_markdown"
                    # Fallback to raw_markdown
                    elif hasattr(md, 'raw_markdown') and md.raw_markdown:
                        content = md.raw_markdown
                        content_source = "raw_markdown"
                
                # If no markdown, try extracted_content
                if not content and hasattr(doc, 'extracted_content') and doc.extracted_content:
                    content = doc.extracted_content
                    content_source = "extracted_content"
                
                # Validate we have content
                if not content:
                    print(f"    [{i}] ✗ No content found for {url[:50]}", flush=True)
                    continue
                
                # Convert to string and validate length
                content_str = str(content).strip()
                if len(content_str) < 50:
                    print(f"    [{i}] ✗ Content too short ({len(content_str)} chars)", flush=True)
                    continue
                
                # Add to pages
                pages.append({
                    'url': url,
                    'content': content_str,
                    'score': 0.5,
                    'content_source': content_source
                })
                print(f"    [{i}] ✓ {len(content_str):>6} chars from {content_source:>15} - {url[:50]}", flush=True)
            
            except Exception as e:
                print(f"    [{i}] ✗ Error processing doc: {e}", flush=True)
                import traceback
                traceback.print_exc()
        
        # Summary
        total_chars = sum(len(p.get('content', '')) for p in pages)
        print(f"  ✅ Successfully extracted {len(pages)} pages ({total_chars:,} chars total)", flush=True)
    
    except Exception as e:
        print(f"  ❌ Exception in extract_pages_from_result: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    print(f"<<< Returning {len(pages)} pages\n", flush=True)
    
    return pages


@app.get("/")
async def root():
    """Service status endpoint."""
    openrouter_configured = bool(os.environ.get("OPENROUTER_API_KEY"))
    return {
        "message": "Crawl4AI Adaptive Crawler v3.7.1 (SPA + EXTRACTION FIXES) is running!",
        "version": "3.7.1",
        "status": "✅ Ready",
        "adaptive_available": ADAPTIVE_AVAILABLE,
        "deep_crawl_fallback": DEEP_CRAWL_AVAILABLE,
        "openrouter_reranking": openrouter_configured,
        "embedding_model_verified": EMBEDDING_MODEL_VERIFIED,
        "fixes_v3.7.1": [
            "✓ FIX: SPAFriendlyCrawler wrapper for JavaScript/SPA sites",
            "✓ FIX: Automatic CrawlerRunConfig injection to AdaptiveCrawler",
            "✓ FIX: wait_for='css:body' ensures content renders",
            "✓ FIX: process_iframes=True includes embedded content",
            "✓ FIX: delay_before_return_html=2.0s for JS execution",
            "✓ FIX: Embedding model verification at startup",
            "✓ FIX: Verbose logging for debugging",
            "✓ FIX 4: CrawlResult.markdown extraction (was looking for .content instead of .markdown.raw_markdown)",
            "✓ FIX 5: Format string bugs (:.0% formatter with manual multiplication)",
            "✓ FIX 6: Early validation of extracted content - fail fast if KB extraction fails"
        ],
        "previous_solutions": [
            "✓ Aggressive stopping prevention (confidence=0.05)",
            "✓ Better embedding model (mpnet multilingual)",
            "✓ Better re-ranking weights (75/25)",
            "✓ More query variations (20)",
            "✓ Deeper crawling (50 pages, 25 links)",
            "✓ Retry logic for API calls"
        ],
        "features": [
            "LOCAL multilingual embedding strategy",
            "SPA/JavaScript site support via SPAFriendlyCrawler",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "Chinese language support",
            "OpenRouter qwen3-embedding-8b (optional)",
            "Retry logic with exponential backoff",
            "Aggressive exploration prevents early termination",
            "DeepSeek-reasoner for answer generation"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/crawl", response_model=CrawlResponse)
async def adaptive_crawl(request: CrawlRequest):
    """Perform adaptive crawling and return a direct answer."""

    # Check for required API keys
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY environment variable not set"
        )

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    use_embeddings = request.use_embeddings and bool(openrouter_api_key)

    if request.use_embeddings and not openrouter_api_key:
        print("Warning: OPENROUTER_API_KEY not set, skipping embedding re-ranking", flush=True)

    try:
        # Extract domain from start URL
        parsed_url = urlparse(request.start_url)
        domain = parsed_url.netloc

        # Detect languages in query
        lang_info = detect_query_languages(request.query)

        print(f"\n{'='*60}", flush=True)
        print(f"Query: {request.query}", flush=True)
        print(f"Query Languages: {lang_info['languages']}", flush=True)
        print(f"Start URL: {request.start_url}", flush=True)
        print(f"Domain: {domain}", flush=True)
        print(f"Max pages: {request.max_pages}", flush=True)
        print(f"Confidence threshold: {request.confidence_threshold}", flush=True)
        print(f"Embedding re-ranking: {use_embeddings}", flush=True)
        if use_embeddings:
            print(f"Embedding model: {request.embedding_model}", flush=True)
        print(f"{'='*60}", flush=True)

        # Create embedding client if available
        embedding_client = None
        if use_embeddings:
            embedding_client = OpenRouterEmbeddings(
                api_key=openrouter_api_key,
                model=request.embedding_model
            )

        if ADAPTIVE_AVAILABLE:
            return await run_adaptive_crawl(
                request=request,
                deepseek_api_key=deepseek_api_key,
                embedding_client=embedding_client,
                domain=domain
            )
        elif DEEP_CRAWL_AVAILABLE:
            return await run_fallback_deep_crawl(
                request=request,
                deepseek_api_key=deepseek_api_key,
                embedding_client=embedding_client,
                domain=domain
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Neither AdaptiveCrawler nor deep crawling is available"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def run_adaptive_crawl(
    request: CrawlRequest,
    deepseek_api_key: str,
    embedding_client: Optional[OpenRouterEmbeddings],
    domain: str
) -> CrawlResponse:
    """Run crawl using AdaptiveCrawler with LOCAL EMBEDDING strategy + OpenRouter re-ranking."""

    if request.use_embeddings:
        print("Using EMBEDDING strategy for semantic link selection", flush=True)
        print("  ✓ SOLUTION 2: Model: paraphrase-multilingual-mpnet-base-v2", flush=True)
        print("  ✓ SOLUTION 1: Aggressive exploration: confidence_threshold=0.05", flush=True)
        print("  ✓ SOLUTION 4: Query variations: 20", flush=True)
        print("  ✓ SOLUTION 5: max_pages=50, top_k_links=25", flush=True)

        config = AdaptiveConfig(
            strategy="embedding",
            embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            confidence_threshold=0.05,
            embedding_min_confidence_threshold=0.02,
            embedding_validation_min_score=0.1,
            embedding_min_relative_improvement=0.01,
            n_query_variations=20,
            max_pages=request.max_pages or 50,
            top_k_links=25,
            embedding_coverage_radius=0.35,
            embedding_k_exp=1.5,
            embedding_overlap_threshold=0.80,
        )
        used_embedding_crawl = True
    else:
        print("Using STATISTICAL (BM25) strategy", flush=True)
        config = AdaptiveConfig(
            strategy="statistical",
            confidence_threshold=0.05,
            max_pages=request.max_pages or 50,
            top_k_links=25,
            min_gain_threshold=0.01
        )
        used_embedding_crawl = False

    browser_config = BrowserConfig(
        headless=True,
        verbose=True,  # Enable verbose for debugging
        java_script_enabled=True,  # Ensure JS is enabled for SPA sites
    )

    # FIX: Use SPAFriendlyCrawler wrapper to automatically inject CrawlerRunConfig
    # This solves the root cause: AdaptiveCrawler doesn't accept CrawlerRunConfig,
    # so SPA sites (like macaolaws.zeabur.app) don't render properly.
    async with AsyncWebCrawler(config=browser_config) as raw_crawler:
        # Wrap the crawler with SPA-friendly defaults
        crawler = SPAFriendlyCrawler(raw_crawler)
        adaptive = AdaptiveCrawler(crawler, config=config)

        print("Starting adaptive crawl with AGGRESSIVE EXPLORATION...", flush=True)
        print(f"  SPAFriendlyCrawler will inject: wait_for, process_iframes, delay", flush=True)
        result = await adaptive.digest(
            start_url=request.start_url,
            query=request.query
        )

        # VERBOSE DEBUG: Log crawl result details
        print(f"\n{'='*40} CRAWL RESULT DEBUG {'='*40}", flush=True)
        print(f"Result type: {type(result).__name__}", flush=True)
        
        if result:
            # Safely handle crawled_urls (it's a set, not a list)
            if hasattr(result, 'crawled_urls'):
                crawled_urls_count = len(result.crawled_urls)
                print(f"✓ Crawled URLs count: {crawled_urls_count}", flush=True)
                crawled_urls_list = list(result.crawled_urls)[:5]
                print(f"  Sample URLs: {crawled_urls_list}", flush=True)
            else:
                crawled_urls_count = 0
            
            # Check knowledge_base
            if hasattr(result, 'knowledge_base'):
                kb = result.knowledge_base
                print(f"✓ Knowledge base exists: {kb is not None}", flush=True)
                
                if kb:
                    kb_type = type(kb).__name__
                    kb_len = len(kb) if hasattr(kb, '__len__') else 'unknown'
                    print(f"  Type: {kb_type}, Length: {kb_len}", flush=True)
                    
                    if isinstance(kb, list) and len(kb) > 0:
                        print(f"  KB[0] type: {type(kb[0]).__name__}", flush=True)
                        kb_item = kb[0]
                        
                        # Inspect first item
                        if isinstance(kb_item, dict):
                            print(f"    Dict keys: {list(kb_item.keys())}", flush=True)
                        else:
                            # It's an object (CrawlResult)
                            obj_type = type(kb_item).__name__
                            print(f"    Object type: {obj_type}", flush=True)
                            
                            # Check for .markdown attribute
                            if hasattr(kb_item, 'markdown'):
                                md = kb_item.markdown
                                print(f"    Has .markdown: {md is not None}", flush=True)
                                if md:
                                    md_type = type(md).__name__
                                    print(f"      .markdown type: {md_type}", flush=True)
                                    
                                    for attr in ['raw_markdown', 'fit_markdown', 'fit_html']:
                                        if hasattr(md, attr):
                                            val = getattr(md, attr, None)
                                            if val:
                                                val_len = len(str(val))
                                                print(f"        .{attr}: {val_len} chars", flush=True)
                else:
                    print(f"  ⚠️ Knowledge base is EMPTY (None)", flush=True)
            
            # Check term extraction
            if hasattr(result, 'term_frequencies'):
                tf = result.term_frequencies
                if tf:
                    print(f"✓ Term frequencies: {len(tf)} terms found", flush=True)
                    top_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"  Top terms: {top_terms}", flush=True)
                else:
                    print(f"⚠️ Term frequencies: EMPTY (0 terms)", flush=True)
            
            # Check documents with terms
            if hasattr(result, 'documents_with_terms'):
                dwt = result.documents_with_terms
                if dwt:
                    print(f"✓ Documents with terms: {len(dwt)} documents", flush=True)
                else:
                    print(f"⚠️ Documents with terms: EMPTY", flush=True)
            
            # Check metrics
            if hasattr(result, 'metrics'):
                metrics = result.metrics
                if metrics:
                    print(f"✓ Metrics: {metrics}", flush=True)
        
        print(f"{'='*40} END DEBUG {'='*40}\n", flush=True)

        # FIX: Handle None/empty result
        if not result:
            return CrawlResponse(
                success=False,
                answer="Crawl returned no result.",
                confidence=0.0,
                pages_crawled=0,
                sources=[],
                message="Crawling failed - no result returned",
                embedding_used=False
            )

        # Get crawl statistics
        confidence = adaptive.confidence if hasattr(adaptive, 'confidence') else 0.0
        crawled_urls = result.crawled_urls if hasattr(result, 'crawled_urls') else []
        pages_crawled = len(crawled_urls)  # ✅ This works because len() works on sets too

        print(f"Crawl complete: {pages_crawled} pages, {int(confidence * 100)}% confidence", flush=True)

        if hasattr(adaptive, 'print_stats'):
            adaptive.print_stats()

        if pages_crawled == 0:
            return CrawlResponse(
                success=False,
                answer="No pages could be crawled from the provided URL.",
                confidence=0.0,
                pages_crawled=0,
                sources=[],
                message="Crawling failed - no pages found",
                embedding_used=False
            )

        # FIX: Extract pages from knowledge base
        relevant_pages = extract_pages_from_result(result)
        
        # Early validation: if no pages extracted, fail fast
        if not relevant_pages:
            print(f"\n⚠️  CRITICAL: No pages extracted from knowledge base!", flush=True)
            print(f"   Crawled URLs: {pages_crawled}", flush=True)
            if hasattr(result, 'knowledge_base'):
                kb_size = len(result.knowledge_base) if result.knowledge_base else 0
                print(f"   Knowledge base size: {kb_size}", flush=True)
            
            return CrawlResponse(
                success=False,
                answer="Pages were crawled but content extraction failed. This may indicate an issue with the target website's structure or content encoding.",
                confidence=confidence,
                pages_crawled=pages_crawled,
                sources=[],
                message=f"EXTRACTION FAILURE: Crawled {pages_crawled} pages but extracted 0 pages from knowledge base",
                embedding_used=False
            )

        strategy_name = "Embedding" if used_embedding_crawl else "BM25"
        print(f"\n{strategy_name} top pages (before re-ranking):", flush=True)
        for i, page in enumerate(relevant_pages[:5], 1):
            # ✅ FIXED: Proper format string
            score_pct = int(page.get('score', 0) * 100)
            print(f"  {i}. {score_pct}% - {page.get('url', 'unknown')}", flush=True)

        # Re-rank with OpenRouter embeddings if available
        embedding_used = False
        if embedding_client and relevant_pages:
            try:
                print(f"\nRe-ranking with OpenRouter...", flush=True)
                relevant_pages = await rerank_by_embeddings(
                    query=request.query,
                    pages=relevant_pages,
                    embedding_client=embedding_client,
                    top_k=15
                )
                embedding_used = True

                print(f"After OpenRouter re-ranking:", flush=True)
                for i, page in enumerate(relevant_pages[:5], 1):
                    emb_score = page.get('embedding_score', 0)
                    # ✅ FIXED: Proper format string without double multiplication
                    combined_pct = int(page['score'] * 100)
                    emb_pct = int(emb_score * 100)
                    print(f"  {i}. Combined: {combined_pct}% (Semantic: {emb_pct}%) - {page['url']}", flush=True)
            except Exception as e:
                print(f"OpenRouter re-ranking failed: {e}, using crawl results", flush=True)
                relevant_pages = relevant_pages[:15]
        else:
            relevant_pages = relevant_pages[:15]

        # Format context for DeepSeek  
        context = format_adaptive_context(relevant_pages)  
        
        if not context.strip():  
            print(f"\n⚠️  WARNING: Formatted context is empty!", flush=True)
            return CrawlResponse(  
                success=False,  
                answer="Pages were crawled and extracted but contain insufficient readable content.",  
                confidence=confidence,  
                pages_crawled=pages_crawled,  
                sources=[],  
                message="CONTENT VALIDATION FAILURE: Context formatting produced empty result",  
                embedding_used=embedding_used  
            )  
        
        print(f"\nContext length: {len(context):,} chars", flush=True)  
        print("Generating answer with DeepSeek-reasoner...", flush=True)  
        
        answer = await call_deepseek(  
            query=request.query,  
            context=context,  
            api_key=deepseek_api_key  
        )  
        
        # Prepare sources with deduplication  
        sources = []  
        seen_urls = set()  
        for page in relevant_pages[:15]:  
            url = page.get('url', 'unknown')  
            if url in seen_urls:  
                continue  
            seen_urls.add(url)  
            source_info = {  
                "url": url,  
                "relevance": round(page.get('score', 0), 3)  
            }  
            if embedding_used and 'embedding_score' in page:  
                source_info["semantic_score"] = round(page.get('embedding_score', 0), 3)  
            sources.append(source_info)  
        
        crawl_strategy = "embedding" if used_embedding_crawl else "statistical"  
        
        # ✅ FIXED: Proper format string
        return CrawlResponse(  
            success=True,  
            answer=answer,  
            confidence=round(confidence, 3),  
            pages_crawled=pages_crawled,  
            sources=sources,  
            message=(  
                f"Adaptive crawl ({crawl_strategy}): {pages_crawled} pages, {int(confidence*100)}% confidence" +  
                (" (with OpenRouter re-ranking)" if embedding_used else "")  
            ),  
            embedding_used=embedding_used or used_embedding_crawl  
        )  


async def run_fallback_deep_crawl(
    request: CrawlRequest,
    deepseek_api_key: str,
    embedding_client: Optional[OpenRouterEmbeddings],
    domain: str
) -> CrawlResponse:
    """Fallback to deep crawling if AdaptiveCrawler is not available."""

    print("Using FALLBACK deep crawl (BestFirst strategy)", flush=True)

    domain_filter = DomainFilter(
        allowed_domains=[domain],
        blocked_domains=[]
    )

    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'what', 'which', 'who', 'how', 'when', 'where', 'why', 'i', 'you',
        'show', 'get', 'find', 'give', 'me', 'my', 'your', 'want', 'need',
        '我', '是', '在', '会', '被', '了', '吗', '嘛'
    }
    keywords = [
        word.lower() for word in request.query.split()
        if len(word) > 2 and word.lower() not in stop_words
    ]

    print(f"Keywords: {keywords}", flush=True)

    url_scorer = KeywordRelevanceScorer(
        keywords=keywords,
        weight=0.9
    )

    strategy = BestFirstCrawlingStrategy(
        max_depth=4,
        include_external=False,
        max_pages=request.max_pages or 50,
        filter_chain=FilterChain([domain_filter]),
        url_scorer=url_scorer
    )

    crawl_config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        stream=False,
        verbose=True
    )

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        print("Starting deep crawl with SOLUTION 5: Deeper exploration...", flush=True)
        results = await crawler.arun(
            url=request.start_url,
            config=crawl_config
        )

        if isinstance(results, list):
            crawled_pages = results
        else:
            crawled_pages = [results] if results else []

        successful_pages = [r for r in crawled_pages if hasattr(r, 'success') and r.success]
        pages_crawled = len(successful_pages)

        print(f"Crawled {pages_crawled} pages successfully", flush=True)

        if not successful_pages:
            return CrawlResponse(
                success=False,
                answer="No content could be crawled from the provided URL.",
                confidence=0.0,
                pages_crawled=0,
                sources=[],
                message="Crawling failed - no pages found",
                embedding_used=False
            )

        pages_for_ranking = []
        for result in successful_pages:
            url = result.url if hasattr(result, 'url') else str(result)
            content = ""
            
            # FIX: Extract from .markdown properly
            if hasattr(result, 'markdown') and result.markdown:
                md = result.markdown
                if hasattr(md, 'fit_markdown') and md.fit_markdown:
                    content = md.fit_markdown
                elif hasattr(md, 'raw_markdown') and md.raw_markdown:
                    content = md.raw_markdown
            
            # Fallback
            if not content and hasattr(result, 'extracted_content') and result.extracted_content:
                content = result.extracted_content

            if content and len(content) >= 50:
                pages_for_ranking.append({
                    'url': url,
                    'content': content,
                    'score': 0.5
                })

        embedding_used = False
        if embedding_client and pages_for_ranking:
            try:
                pages_for_ranking = await rerank_by_embeddings(
                    query=request.query,
                    pages=pages_for_ranking,
                    embedding_client=embedding_client,
                    top_k=15
                )
                embedding_used = True
            except Exception as e:
                print(f"Embedding re-ranking failed: {e}", flush=True)
                pages_for_ranking = pages_for_ranking[:15]
        else:
            pages_for_ranking = pages_for_ranking[:15]

        context = format_adaptive_context(pages_for_ranking)

        if not context.strip():
            return CrawlResponse(
                success=False,
                answer="Pages were crawled but no readable content was extracted.",
                confidence=0.0,
                pages_crawled=pages_crawled,
                sources=[],
                message="No content extracted",
                embedding_used=embedding_used
            )

        print(f"Context length: {len(context):,} chars", flush=True)
        print("Generating answer with DeepSeek-reasoner...", flush=True)

        answer = await call_deepseek(
            query=request.query,
            context=context,
            api_key=deepseek_api_key
        )

        sources = []
        seen_urls = set()
        for page in pages_for_ranking[:15]:
            url = page.get('url', 'unknown')
            if url in seen_urls:
                continue
            seen_urls.add(url)

            source_info = {
                "url": url,
                "relevance": round(page.get('score', 0), 3)
            }
            if embedding_used and 'embedding_score' in page:
                source_info["semantic_score"] = round(page.get('embedding_score', 0), 3)
            sources.append(source_info)

        # ✅ FIXED: Proper format string
        return CrawlResponse(
            success=True,
            answer=answer,
            confidence=0.5,
            pages_crawled=pages_crawled,
            sources=sources,
            message=(
                f"Deep crawl complete (SOLUTION 5 applied): {pages_crawled} pages (fallback mode)" +
                (" (with OpenRouter re-ranking)" if embedding_used else "")
            ),
            embedding_used=embedding_used
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server on port {port}...", flush=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
