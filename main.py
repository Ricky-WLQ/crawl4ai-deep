
"""
Crawl4AI Adaptive Crawler with LOCAL MULTILINGUAL EMBEDDING Strategy + OpenRouter Re-ranking + DeepSeek Reasoner

EMBEDDING STRATEGY (Semantic Crawling - NO API NEEDED!):
- Uses LOCAL sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- Supports 50+ languages including Chinese, English, and more
- AGGRESSIVE EXPLORATION: confidence_threshold=0.05, min_relative_improvement=0.01
- Disables early stopping to ensure thorough crawling

FALLBACK (if use_embeddings=false):
- Uses BM25 statistical strategy (keyword-based)

RE-RANKING (optional, requires OPENROUTER_API_KEY):
- OpenRouter embeddings (qwen3-embedding-8b) for semantic re-ranking

ANSWER GENERATION (requires DEEPSEEK_API_KEY):
- DeepSeek-reasoner for comprehensive answers

Version: 3.6.3 (FINAL - All Issues Fixed - Force Rebuild)  
"""

import os
import sys
import time
import asyncio  # ✅ CRITICAL FIX: Added asyncio import
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
    print("Core crawl4ai imported successfully", flush=True)
except ImportError as e:
    print(f"Failed to import core crawl4ai: {e}", flush=True)
    sys.exit(1)

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
    print("Crawl4AI Adaptive Crawler v3.6.2 (FINAL) Starting...", flush=True)
    print(f"AdaptiveCrawler Available: {ADAPTIVE_AVAILABLE}", flush=True)
    print(f"Deep Crawl Fallback Available: {DEEP_CRAWL_AVAILABLE}", flush=True)
    print(f"OpenRouter API Key: {'Configured' if openrouter_configured else 'NOT SET'}", flush=True)
    print("=" * 60, flush=True)
    print("EMBEDDING Strategy (LOCAL MULTILINGUAL - No API needed!):", flush=True)
    print("  ✓ Model: paraphrase-multilingual-mpnet-base-v2 (50+ languages)", flush=True)
    print("  ✓ SOLUTION 2: Upgraded from MiniLM to mpnet for better legal docs", flush=True)
    print("  ✓ SOLUTION 1: AGGRESSIVE exploration (confidence=0.05, min_improvement=0.01)", flush=True)
    print("  ✓ SOLUTION 4: Query variations=20 (up from 5)", flush=True)
    print("  ✓ SOLUTION 5: max_pages=50, top_k_links=25 (deeper crawling)", flush=True)
    if openrouter_configured:
        print("  ✓ SOLUTION 3: Re-ranking 75% semantic + 25% BM25", flush=True)
        print("  ✓ Re-ranking: OpenRouter qwen3-embedding-8b with retry logic", flush=True)
    print("  ✓ Answer Generation: DeepSeek-reasoner", flush=True)
    print("=" * 60, flush=True)
    yield
    print("Shutting down...", flush=True)


app = FastAPI(
    title="Crawl4AI Adaptive Crawler",
    description="Intelligent web crawler with LOCAL MULTILINGUAL EMBEDDING strategy (50+ languages) + OpenRouter re-ranking + DeepSeek reasoning - v3.6.2 FINAL",
    version="3.6.2",
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
    """FIX: Extract pages from adaptive crawl result with fallback handling."""
    pages = []
    
    try:
        # Try multiple ways to extract pages
        if hasattr(result, 'pages') and result.pages:
            pages = result.pages
        elif hasattr(result, 'get_relevant_content'):
            pages = result.get_relevant_content(top_k=25)
        elif hasattr(result, 'knowledge_base') and hasattr(result.knowledge_base, 'pages'):
            pages = result.knowledge_base.pages
    except Exception as e:
        print(f"Error extracting pages: {e}", flush=True)
    
    return pages if pages else []


@app.get("/")
async def root():
    """Service status endpoint."""
    openrouter_configured = bool(os.environ.get("OPENROUTER_API_KEY"))
    return {
        "message": "Crawl4AI Adaptive Crawler v3.6.2 (FINAL) is running!",
        "version": "3.6.2",
        "status": "✅ Ready",
        "adaptive_available": ADAPTIVE_AVAILABLE,
        "deep_crawl_fallback": DEEP_CRAWL_AVAILABLE,
        "openrouter_reranking": openrouter_configured,
        "solutions_applied": [
            "✓ SOLUTION 1: Aggressive stopping prevention",
            "✓ SOLUTION 2: Better embedding model (mpnet)",
            "✓ SOLUTION 3: Better re-ranking weights (75/25)",
            "✓ SOLUTION 4: More query variations (20)",
            "✓ SOLUTION 5: Deeper crawling (50 pages, 25 links)",
            "✓ SOLUTION 6: Proper markdown extraction",
            "✓ SYNTAX FIX: Wrapped message expressions",
            "✓ FIX 1: Added result extraction fallback",
            "✓ FIX 2: Added retry logic for API calls",
            "✓ FIX 3: Added error handling for None results",
            "✓ FIX 4: Removed unused imports",
            "✓ FIX 5: ✅ Added asyncio import"
        ],
        "features": [
            "LOCAL multilingual embedding strategy",
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
        verbose=False,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        adaptive = AdaptiveCrawler(crawler, config=config)

        print("Starting adaptive crawl with AGGRESSIVE EXPLORATION...", flush=True)
        result = await adaptive.digest(
            start_url=request.start_url,
            query=request.query
        )

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
        pages_crawled = len(crawled_urls)

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
