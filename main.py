"""
Crawl4AI Adaptive Crawler with LOCAL MULTILINGUAL EMBEDDING Strategy + OpenRouter Re-ranking + DeepSeek Reasoner

EMBEDDING STRATEGY (Semantic Crawling - NO API NEEDED!):
- Uses LOCAL sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Supports 50+ languages including Chinese, English, and more
- Optimized stopping criteria to prevent early termination
- Semantic link selection understands meaning, not just keywords

FALLBACK (if use_embeddings=false):
- Uses BM25 statistical strategy (keyword-based)

RE-RANKING (optional, requires OPENROUTER_API_KEY):
- OpenRouter embeddings (qwen3-embedding-8b) for semantic re-ranking

ANSWER GENERATION (requires DEEPSEEK_API_KEY):
- DeepSeek-reasoner for comprehensive answers

Version: 3.5.1
"""

import os
import sys
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
    from crawl4ai import AdaptiveCrawler, AdaptiveConfig, LLMConfig
    ADAPTIVE_AVAILABLE = True
    print("AdaptiveCrawler imported successfully", flush=True)
except ImportError as e:
    print(f"AdaptiveCrawler not available: {e}", flush=True)
    ADAPTIVE_AVAILABLE = False

# Check if LLMConfig is available (for native embedding strategy)
LLMCONFIG_AVAILABLE = 'LLMConfig' in dir()
if not LLMCONFIG_AVAILABLE:
    try:
        from crawl4ai import LLMConfig
        LLMCONFIG_AVAILABLE = True
    except ImportError:
        print("LLMConfig not available, will use fallback embedding", flush=True)
        LLMCONFIG_AVAILABLE = False

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
# Custom OpenRouter Embedding Client
# ============================================================================

class OpenRouterEmbeddings:
    """Custom client for OpenRouter embeddings API."""

    def __init__(self, api_key: str, model: str = "qwen/qwen3-embedding-8b"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/embeddings"

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts from OpenRouter."""
        if not texts:
            return []

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

            if response.status_code != 200:
                print(f"OpenRouter embedding error: {response.status_code} - {response.text}", flush=True)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"OpenRouter embedding error: {response.text}"
                )

            result = response.json()
            # Extract embeddings in order
            embeddings = [None] * len(texts)
            for item in result.get("data", []):
                idx = item.get("index", 0)
                embeddings[idx] = item.get("embedding", [])

            return embeddings

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0

    a = np.array(vec1)
    b = np.array(vec2)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


async def rerank_by_embeddings(
    query: str,
    pages: List[dict],
    embedding_client: OpenRouterEmbeddings,
    top_k: int = 10
) -> List[dict]:
    """Re-rank pages by semantic similarity to query using embeddings."""

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

        # Combine with original BM25 score if available
        original_score = page.get('score', 0.5)
        # Weighted combination: 60% embedding, 40% BM25
        combined_score = (0.6 * similarity) + (0.4 * original_score)

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
    print("Crawl4AI Adaptive Crawler v3.5.1 Starting...", flush=True)
    print(f"AdaptiveCrawler Available: {ADAPTIVE_AVAILABLE}", flush=True)
    print(f"Deep Crawl Fallback Available: {DEEP_CRAWL_AVAILABLE}", flush=True)
    print(f"OpenRouter API Key: {'Configured' if openrouter_configured else 'NOT SET'}", flush=True)
    print("=" * 60, flush=True)
    print("EMBEDDING Strategy (LOCAL MULTILINGUAL - No API needed!):", flush=True)
    print("  - Model: paraphrase-multilingual-MiniLM-L12-v2 (50+ languages)", flush=True)
    print("  - Optimized stopping criteria for thorough crawling", flush=True)
    if openrouter_configured:
        print("  - Re-ranking: OpenRouter qwen3-embedding-8b", flush=True)
    print("  - Answer Generation: DeepSeek-reasoner", flush=True)
    print("=" * 60, flush=True)
    yield
    print("Shutting down...", flush=True)


app = FastAPI(
    title="Crawl4AI Adaptive Crawler",
    description="Intelligent web crawler with LOCAL MULTILINGUAL EMBEDDING strategy (50+ languages) + OpenRouter re-ranking + DeepSeek reasoning",
    version="3.5.1",
    lifespan=lifespan
)


class CrawlRequest(BaseModel):
    """Request model for adaptive crawling."""
    start_url: str
    query: str
    max_pages: Optional[int] = 20
    confidence_threshold: Optional[float] = 0.7
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
    max_tokens: int = 3000
) -> str:
    """Call DeepSeek API to generate an answer."""

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

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"DeepSeek API error: {response.text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]


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

        # Show both scores if embedding was used
        if embedding_score is not None:
            score_info = f"Combined: {score:.0%}, Semantic: {embedding_score:.0%}"
        else:
            score_info = f"Relevance: {score:.0%}"

        entry = f"""
=== Page {i} ({score_info}): {url} ===
{page_text}

"""
        if total_chars + len(entry) > max_chars:
            break

        context_parts.append(entry)
        total_chars += len(entry)

    return "\n".join(context_parts)


@app.get("/")
async def root():
    """Service status endpoint."""
    openrouter_configured = bool(os.environ.get("OPENROUTER_API_KEY"))
    return {
        "message": "Crawl4AI Adaptive Crawler is running!",
        "version": "3.5.1",
        "adaptive_available": ADAPTIVE_AVAILABLE,
        "deep_crawl_fallback": DEEP_CRAWL_AVAILABLE,
        "openrouter_reranking": openrouter_configured,
        "features": [
            "LOCAL multilingual embedding strategy (no API needed for crawling)",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (50+ languages)",
            "OpenRouter qwen3-embedding-8b for re-ranking (optional)",
            "Optimized stopping criteria to prevent early termination",
            "Increased link exploration (top_k=15)",
            "DeepSeek-reasoner for answer generation"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/crawl", response_model=CrawlResponse)
async def adaptive_crawl(request: CrawlRequest):
    """
    Perform adaptive crawling and return a direct answer.

    - Uses AdaptiveCrawler with EMBEDDING strategy for semantic link selection
    - OpenRouter embeddings (qwen3-embedding-8b) for crawling and re-ranking
    - Increased link exploration (top_k=15) to discover more relevant content
    - DeepSeek-reasoner generates comprehensive answer
    """

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
                domain=domain,
                openrouter_api_key=openrouter_api_key
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
    domain: str,
    openrouter_api_key: Optional[str] = None
) -> CrawlResponse:
    """Run crawl using AdaptiveCrawler with LOCAL EMBEDDING strategy + OpenRouter re-ranking.

    EMBEDDING STRATEGY (LOCAL - No API needed!):
    - Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings
    - Runs entirely on the server

    FALLBACK (when use_embeddings=false):
    - BM25 statistical strategy (keyword-based)

    Re-ranking: OpenRouter qwen3-embedding-8b (if OPENROUTER_API_KEY set)
    """

    if request.use_embeddings:
        # Use EMBEDDING strategy with LOCAL sentence-transformers
        # No API key needed for embeddings - runs locally on server!
        print("Using EMBEDDING strategy for semantic link selection", flush=True)
        print("  Embeddings: LOCAL sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", flush=True)

        config = AdaptiveConfig(
            strategy="embedding",
            # Use MULTILINGUAL model for Chinese/English support (50+ languages)
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            # DO NOT set embedding_llm_config - it overrides local model!
            confidence_threshold=request.confidence_threshold,
            max_pages=request.max_pages,
            top_k_links=15,  # Increased from 5 to explore more links
            min_gain_threshold=0.01,  # Lowered to avoid early stopping
            n_query_variations=5,  # Query variations
            # CRITICAL: Lower stopping thresholds to prevent early stopping
            embedding_min_relative_improvement=0.01,  # Default 0.1 was too high
            embedding_validation_min_score=0.05,      # Default 0.3 was too high
            embedding_min_confidence_threshold=0.01,  # Default 0.1 was too high
        )
        used_embedding_crawl = True
    else:
        # Fallback to statistical strategy if use_embeddings=false
        print("Using STATISTICAL (BM25) strategy", flush=True)
        config = AdaptiveConfig(
            strategy="statistical",
            confidence_threshold=request.confidence_threshold,
            max_pages=request.max_pages,
            top_k_links=15,  # Increased from 5
            min_gain_threshold=0.01  # Lowered
        )
        used_embedding_crawl = False

    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        adaptive = AdaptiveCrawler(crawler, config=config)

        print("Starting adaptive crawl...", flush=True)
        result = await adaptive.digest(
            start_url=request.start_url,
            query=request.query
        )

        # Get crawl statistics
        confidence = adaptive.confidence
        crawled_urls = result.crawled_urls if hasattr(result, 'crawled_urls') else []
        pages_crawled = len(crawled_urls)

        print(f"Crawl complete: {pages_crawled} pages, {confidence:.0%} confidence", flush=True)

        # Print stats (if method exists)
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

        # Get relevant content from knowledge base
        relevant_pages = adaptive.get_relevant_content(top_k=20)  # Get more for re-ranking

        # Determine which crawl strategy was used (no longer depends on OpenRouter)
        used_embedding_crawl = LLMCONFIG_AVAILABLE and request.use_embeddings
        strategy_name = "Embedding" if used_embedding_crawl else "BM25"
        print(f"\n{strategy_name} top pages:", flush=True)
        for i, page in enumerate(relevant_pages[:5], 1):
            print(f"  {i}. {page['score']:.0%} - {page['url']}", flush=True)

        # Re-rank with OpenRouter embeddings if available
        embedding_used = False
        if embedding_client and relevant_pages:
            try:
                print(f"\nRe-ranking with OpenRouter ({request.embedding_model})...", flush=True)
                relevant_pages = await rerank_by_embeddings(
                    query=request.query,
                    pages=relevant_pages,
                    embedding_client=embedding_client,
                    top_k=10
                )
                embedding_used = True

                print(f"After OpenRouter re-ranking:", flush=True)
                for i, page in enumerate(relevant_pages[:5], 1):
                    emb_score = page.get('embedding_score', 0)
                    print(f"  {i}. Combined: {page['score']:.0%} (Semantic: {emb_score:.0%}) - {page['url']}", flush=True)
            except Exception as e:
                print(f"OpenRouter re-ranking failed: {e}, using crawl results", flush=True)
                relevant_pages = relevant_pages[:10]
        else:
            relevant_pages = relevant_pages[:10]

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
        for page in relevant_pages[:10]:
            url = page.get('url', 'unknown')
            # Skip duplicate URLs
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

        # Determine which strategy was actually used for the message
        crawl_strategy = "embedding" if used_embedding_crawl else "statistical"

        return CrawlResponse(
            success=True,
            answer=answer,
            confidence=round(confidence, 3),
            pages_crawled=pages_crawled,
            sources=sources,
            message=f"Adaptive crawl ({crawl_strategy}): {pages_crawled} pages, {confidence:.0%} confidence" +
                    (" (with OpenRouter re-ranking)" if embedding_used else ""),
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

    # Create domain filter
    domain_filter = DomainFilter(
        allowed_domains=[domain],
        blocked_domains=[]
    )

    # Extract keywords from query
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'what', 'which', 'who', 'how', 'when', 'where', 'why', 'i', 'you',
        'show', 'get', 'find', 'give', 'me', 'my', 'your', 'want', 'need'
    }
    keywords = [
        word.lower() for word in request.query.split()
        if len(word) > 2 and word.lower() not in stop_words
    ]

    print(f"Keywords: {keywords}", flush=True)

    # Create keyword scorer
    url_scorer = KeywordRelevanceScorer(
        keywords=keywords,
        weight=0.9
    )

    # Create BestFirst strategy
    strategy = BestFirstCrawlingStrategy(
        max_depth=3,
        include_external=False,
        max_pages=request.max_pages,
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
        verbose=False
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        print("Starting deep crawl...", flush=True)
        results = await crawler.arun(
            url=request.start_url,
            config=crawl_config
        )

        # Handle results
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

        # Convert to standard format for re-ranking
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

        # Re-rank with embeddings if available
        embedding_used = False
        if embedding_client and pages_for_ranking:
            try:
                pages_for_ranking = await rerank_by_embeddings(
                    query=request.query,
                    pages=pages_for_ranking,
                    embedding_client=embedding_client,
                    top_k=10
                )
                embedding_used = True
            except Exception as e:
                print(f"Embedding re-ranking failed: {e}", flush=True)
                pages_for_ranking = pages_for_ranking[:10]
        else:
            pages_for_ranking = pages_for_ranking[:10]

        # Format context
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

        # Prepare sources with deduplication
        sources = []
        seen_urls = set()
        for page in pages_for_ranking[:10]:
            url = page.get('url', 'unknown')
            # Skip duplicate URLs
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

        return CrawlResponse(
            success=True,
            answer=answer,
            confidence=0.5,
            pages_crawled=pages_crawled,
            sources=sources,
            message=f"Deep crawl complete: {pages_crawled} pages (fallback mode)" +
                    (" (with semantic re-ranking)" if embedding_used else ""),
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
