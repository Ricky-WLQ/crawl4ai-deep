


"""
Crawl4AI Adaptive Crawler with LOCAL MULTILINGUAL EMBEDDING Strategy + OpenRouter Re-ranking + DeepSeek Reasoner

VERSION 3.7.0 - SPA/JavaScript FIX

ROOT CAUSE FIXES (identified via Context7 documentation):
1. SPAFriendlyCrawler wrapper - AdaptiveCrawler doesn't accept CrawlerRunConfig,
   so SPA/JavaScript sites weren't rendering. The wrapper intercepts arun() calls
   and automatically injects CrawlerRunConfig with wait_for, process_iframes, etc.
2. Embedding model verification at startup - detects model loading failures early
3. Verbose logging for debugging - helps trace content extraction issues

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

Version: 3.7.0 (SPA FIX - Context7 Verified)
"""

import os
import sys
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional, List
from urllib.parse import urlparse

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
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"Rate limited. Waiting {wait_time}s before retry...", flush=True)
                        await asyncio.sleep(wait_time)  # ✅ Now works with asyncio import
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
                print(f"Timeout. Waiting {wait_time}s before retry...", flush=True)
                await asyncio.sleep(wait_time)  # ✅ Now works with asyncio import
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
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler."""
    openrouter_configured = bool(os.environ.get("OPENROUTER_API_KEY"))
    print("=" * 60, flush=True)
    print("Crawl4AI Adaptive Crawler v3.7.0 (SPA FIX) Starting...", flush=True)
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

    print("FIXES APPLIED (v3.7.0):", flush=True)
    print("  ✓ FIX 1: SPAFriendlyCrawler wrapper for JavaScript sites", flush=True)
    print("    - Automatically injects CrawlerRunConfig to AdaptiveCrawler", flush=True)
    print("    - wait_for='css:body' ensures content renders", flush=True)
    print("    - process_iframes=True includes embedded content", flush=True)
    print("    - delay_before_return_html=2.0s for JS execution", flush=True)
    print("  ✓ FIX 2: Embedding model verification at startup", flush=True)
    print("  ✓ FIX 3: Verbose logging for debugging", flush=True)
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
    description="Intelligent web crawler with LOCAL MULTILINGUAL EMBEDDING strategy (50+ languages) + SPA/JavaScript support + OpenRouter re-ranking + DeepSeek reasoning - v3.7.0 SPA FIX",
    version="3.7.0",
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
                    print(f"DeepSeek rate limited. Waiting {wait_time}s...", flush=True)
                    await asyncio.sleep(wait_time)  # ✅ Now works with asyncio import
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
            print(f"DeepSeek timeout. Retrying in {wait_time}s...", flush=True)
            await asyncio.sleep(wait_time)  # ✅ Now works with asyncio import

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

        # ✅ FIXED: Changed {score:.0%} to {score*100:.0f}%
        if embedding_score is not None:
            score_info = f"Combined: {score*100:.0f}%, Semantic: {embedding_score*100:.0f}%"
        else:
            score_info = f"Relevance: {score*100:.0f}%"

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
    - result.knowledge_base: List of Document objects with extracted content
    - result.crawled_urls: List of URLs crawled
    - result.documents_with_terms: Dict mapping documents to extracted terms
    """
    pages = []
    
    print(f"\n>>> extract_pages_from_result() called", flush=True)
    
    try:
        # The knowledge_base contains the actual extracted content
        if hasattr(result, 'knowledge_base') and result.knowledge_base:
            kb = result.knowledge_base
            print(f"  ✓ Knowledge base exists with {len(kb)} items", flush=True)
            
            if isinstance(kb, list):
                print(f"  Processing {len(kb)} knowledge base items...", flush=True)
                
                # Each item should have url and content
                for i, doc in enumerate(kb):
                    try:
                        extracted_page = None
                        
                        if isinstance(doc, dict):
                            # Already a dict
                            print(f"    [{i}] Dict with keys: {list(doc.keys())}", flush=True)
                            content = doc.get('content') or doc.get('text') or doc.get('markdown')
                            
                            if content:
                                content_str = str(content).strip()
                                if len(content_str) > 50:
                                extracted_page = {
                                    'url': doc.get('url', f'doc_{i}'),
                                    'content': str(content),
                                    'score': float(doc.get('score', 0.5))
                                }
                                print(f"      ✓ Extracted: {len(extracted_page['content'])} chars from {extracted_page['url'][:60]}", flush=True)
                            else:
                                content_len = len(str(content)) if content else 0
                                print(f"      ✗ Content too short ({content_len} chars) or empty", flush=True)
                        
                        elif hasattr(doc, '__dict__'):
                            # Object with attributes
                            doc_type = type(doc).__name__
                            print(f"    [{i}] Object type: {doc_type}", flush=True)
                            
                            # Try multiple content attributes
                            content = None
                            content_attr = None
                            for attr_name in ['content', 'text', 'markdown', 'body', 'data', 'page_content']:
                                if hasattr(doc, attr_name):
                                    val = getattr(doc, attr_name, None)
                                    if val and len(str(val)) > 50:
                                        content = val
                                        content_attr = attr_name
                                        print(f"      Found .{attr_name}: {len(str(val))} chars", flush=True)
                                        break
                            
                            # Try to get URL
                            url = None
                            for url_attr in ['url', 'link', 'page_url', 'source']:
                                if hasattr(doc, url_attr):
                                    url = getattr(doc, url_attr, None)
                                    if url:
                                        print(f"      Found .{url_attr}: {url[:60]}", flush=True)
                                        break
                            
                            if content:
                                extracted_page = {
                                    'url': url or f'doc_{i}',
                                    'content': str(content),
                                    'score': float(getattr(doc, 'score', 0.5))
                                }
                                print(f"      ✓ Extracted: {len(extracted_page['content'])} chars", flush=True)
                            else:
                                attrs = [a for a in dir(doc) if not a.startswith('_')]
                                print(f"      ✗ No content found. Attributes: {attrs[:15]}", flush=True)
                        
                        else:
                            # Fallback: try string representation
                            doc_type = type(doc).__name__
                            print(f"    [{i}] Fallback for {doc_type}, trying str()", flush=True)
                            content = str(doc)
                            if len(content) > 50:
                                extracted_page = {
                                    'url': f'doc_{i}',
                                    'content': content,
                                    'score': 0.5
                                }
                                print(f"      ✓ Extracted: {len(extracted_page['content'])} chars from str()", flush=True)
                            else:
                                print(f"      ✗ str() content too short ({len(content)} chars)", flush=True)
                        
                        if extracted_page:
                            pages.append(extracted_page)
                    
                    except Exception as e:
                        print(f"    [{i}] ERROR: {str(e)}", flush=True)
                        import traceback
                        traceback.print_exc()
            
            else:
                print(f"  ⚠️  KB is not a list, it's {type(kb).__name__}", flush=True)
        
        else:
            print(f"  ⚠️  No knowledge_base or it's empty", flush=True)
        
        # Fallback: Try to use get_relevant_content() method if available
        if not pages and hasattr(result, 'get_relevant_content'):
            print(f"  Trying fallback: get_relevant_content()...", flush=True)
            try:
                relevant = result.get_relevant_content(top_k=25)
                if relevant:
                    print(f"    get_relevant_content() returned {len(relevant)} items", flush=True)
                    for j, item in enumerate(relevant):
                        try:
                            if isinstance(item, dict):
                                content = item.get('content') or item.get('text')
                                if content and len(str(content)) > 50:
                                    pages.append({
                                        'url': item.get('url', f'relevant_{j}'),
                                        'content': str(content),
                                        'score': float(item.get('score', 0.5))
                                    })
                                    print(f"      ✓ Added from get_relevant_content: {len(str(content))} chars", flush=True)
                        except Exception as e:
                            print(f"      Error processing item {j}: {e}", flush=True)
            except Exception as e:
                print(f"    get_relevant_content() failed: {e}", flush=True)
        
        # Last resort: Log what we couldn't extract
        if not pages:
            print(f"\n  ❌ NO PAGES EXTRACTED!", flush=True)
            if hasattr(result, 'crawled_urls') and result.crawled_urls:
                print(f"  ℹ️  But {len(result.crawled_urls)} URLs were crawled:", flush=True)
                for url in result.crawled_urls[:5]:
                    print(f"      - {url}", flush=True)
            
            if hasattr(result, 'documents_with_terms'):
                dwt = result.documents_with_terms
                print(f"  ℹ️  Documents with terms: {len(dwt) if dwt else 0}", flush=True)
            
            print(f"\n  DEBUGGING INFO:", flush=True)
            print(f"    Result type: {type(result).__name__}", flush=True)
            print(f"    Result attributes: {[a for a in dir(result) if not a.startswith('_')][:20]}", flush=True)
            
    except Exception as e:
        print(f"  ❌ EXCEPTION in extract_pages_from_result: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    # Final summary
    if pages:
        print(f"\n  ✅ Successfully extracted {len(pages)} pages", flush=True)
        total_chars = sum(len(p.get('content', '')) for p in pages)
        print(f"     Total content: {total_chars} chars across all pages", flush=True)
    else:
        print(f"\n  ⚠️  FINAL RESULT: 0 pages extracted", flush=True)
    
    print(f"<<< extract_pages_from_result() returning {len(pages)} pages\n", flush=True)
    
    return pages


@app.get("/")
async def root():
    """Service status endpoint."""
    openrouter_configured = bool(os.environ.get("OPENROUTER_API_KEY"))
    return {
        "message": "Crawl4AI Adaptive Crawler v3.7.0 (SPA FIX) is running!",
        "version": "3.7.0",
        "status": "✅ Ready",
        "adaptive_available": ADAPTIVE_AVAILABLE,
        "deep_crawl_fallback": DEEP_CRAWL_AVAILABLE,
        "openrouter_reranking": openrouter_configured,
        "embedding_model_verified": EMBEDDING_MODEL_VERIFIED,
        "fixes_v3.7.0": [
            "✓ FIX: SPAFriendlyCrawler wrapper for JavaScript/SPA sites",
            "✓ FIX: Automatic CrawlerRunConfig injection to AdaptiveCrawler",
            "✓ FIX: wait_for='css:body' ensures content renders",
            "✓ FIX: process_iframes=True includes embedded content",
            "✓ FIX: delay_before_return_html=2.0s for JS execution",
            "✓ FIX: Embedding model verification at startup",
            "✓ FIX: Verbose logging for debugging"
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

        print(f"\n{'='*60}", flush=True)
        print(f"Query: {request.query}", flush=True)
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
        print(f"Result type: {type(result)}", flush=True)
        
        if result:
            # Safely handle crawled_urls (it's a set, not a list)
            if hasattr(result, 'crawled_urls'):
                print(f"✓ Crawled URLs count: {len(result.crawled_urls)}", flush=True)
                crawled_urls_list = list(result.crawled_urls)[:5]
                print(f"  Sample URLs: {crawled_urls_list}", flush=True)
            
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
                            for key in ['content', 'text', 'markdown', 'url']:
                                if key in kb_item:
                                    val = kb_item[key]
                                    val_len = len(str(val)) if val else 0
                                    print(f"      .{key}: {val_len} chars", flush=True)
                        else:
                            # It's an object
                            obj_type = type(kb_item).__name__
                            print(f"    Object type: {obj_type}", flush=True)
                            attrs = [a for a in dir(kb_item) if not a.startswith('_')]
                            print(f"    Attributes: {attrs[:15]}", flush=True)
                            
                            # Check for content
                            for attr in ['content', 'text', 'markdown', 'body']:
                                if hasattr(kb_item, attr):
                                    val = getattr(kb_item, attr, None)
                                    if val:
                                        val_len = len(str(val))
                                        print(f"      .{attr}: {val_len} chars", flush=True)
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

        print(f"Crawl complete: {pages_crawled} pages, {confidence:.0%} confidence", flush=True)

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

        # FIX: Use fallback to extract pages
        relevant_pages = extract_pages_from_result(result)
        if not relevant_pages:
            relevant_pages = []

        strategy_name = "Embedding" if used_embedding_crawl else "BM25"
        print(f"\n{strategy_name} top pages (before re-ranking):", flush=True)
        for i, page in enumerate(relevant_pages[:5], 1):
            # ✅ FIXED: Changed {page.get('score', 0):.0%} to {score_val*100:.0f}%
            score_val = page.get('score', 0)
            print(f"  {i}. {score_val*100:.0f}% - {page.get('url', 'unknown')}", flush=True)

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
                    # ✅ FIXED: Changed {page['score']:.0%} and {emb_score:.0%}
                    combined_score = page['score'] * 100
                    emb_score_pct = emb_score * 100
                    print(f"  {i}. Combined: {combined_score:.0f}% (Semantic: {emb_score_pct:.0f}%) - {page['url']}", flush=True)
            except Exception as e:
                print(f"OpenRouter re-ranking failed: {e}, using crawl results", flush=True)
                relevant_pages = relevant_pages[:15]
        else:
            relevant_pages = relevant_pages[:15]

        # Format context for DeepSeek  
        context = format_adaptive_context(relevant_pages)  
        if not context.strip():  
            return CrawlResponse(  
                success=False,  
                answer="Pages were crawled but no readable content was extracted.",  
                confidence=confidence,  
                pages_crawled=pages_crawled,  
                sources=[],  
                message="No content extracted",  
                embedding_used=embedding_used  
            )  
        print(f"\nContext length: {len(context)} chars", flush=True)  
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
        # ✅ FIXED: Changed {confidence:.0%} to {confidence*100:.0f}%  
        return CrawlResponse(  
            success=True,  
            answer=answer,  
            confidence=round(confidence, 3),  
            pages_crawled=pages_crawled,  
            sources=sources,  
            message=(  
                f"Adaptive crawl ({crawl_strategy}): {pages_crawled} pages, {confidence*100:.0f}% confidence" +  
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
            if hasattr(result, 'markdown') and result.markdown:
                content = result.markdown
            elif hasattr(result, 'text') and result.text:
                content = result.text

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

        print(f"Context length: {len(context)} chars", flush=True)
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

        # ✅ FIXED: Changed {confidence*100:.0f}% format
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
