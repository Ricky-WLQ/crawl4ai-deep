"""
Crawl4AI Adaptive Crawler with OpenRouter Embeddings + DeepSeek Reasoner
- Uses AdaptiveCrawler for intelligent subpage discovery
- OpenRouter API for semantic embeddings (text-embedding-3-small)
- Automatically stops when sufficient information is gathered
- DeepSeek-reasoner for answer generation
"""

import os
import sys
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler."""
    print("=" * 60, flush=True)
    print("Crawl4AI Adaptive Crawler Starting...", flush=True)
    print(f"AdaptiveCrawler Available: {ADAPTIVE_AVAILABLE}", flush=True)
    print(f"Deep Crawl Fallback Available: {DEEP_CRAWL_AVAILABLE}", flush=True)
    print("Embedding: OpenRouter (text-embedding-3-small)", flush=True)
    print("Answer Generation: DeepSeek-reasoner", flush=True)
    print("=" * 60, flush=True)
    yield
    print("Shutting down...", flush=True)


app = FastAPI(
    title="Crawl4AI Adaptive Crawler",
    description="Intelligent web crawler with semantic embeddings and DeepSeek reasoning",
    version="2.0.0",
    lifespan=lifespan
)


class CrawlRequest(BaseModel):
    """Request model for adaptive crawling."""
    start_url: str
    query: str
    max_pages: Optional[int] = 20
    confidence_threshold: Optional[float] = 0.7
    use_embeddings: Optional[bool] = True
    embedding_model: Optional[str] = "openrouter/qwen/qwen3-embedding-8b"  # LiteLLM format: openrouter/<model>


class CrawlResponse(BaseModel):
    """Response model with direct answer."""
    success: bool
    answer: str
    confidence: float
    pages_crawled: int
    sources: List[dict]
    message: str


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
        content = page.get('content', '')

        if not content or len(content) < 50:
            continue

        # Truncate individual page content
        page_text = content[:5000] if len(content) > 5000 else content

        entry = f"""
=== Page {i} (Relevance: {score:.0%}): {url} ===
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
    return {
        "message": "Crawl4AI Adaptive Crawler is running!",
        "version": "2.0.0",
        "adaptive_available": ADAPTIVE_AVAILABLE,
        "deep_crawl_fallback": DEEP_CRAWL_AVAILABLE,
        "features": [
            "AdaptiveCrawler with semantic embeddings",
            "Intelligent subpage discovery",
            "Automatic confidence-based stopping",
            "OpenRouter embeddings (text-embedding-3-small)",
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

    - Uses AdaptiveCrawler for intelligent link discovery
    - Semantic embeddings score pages by content relevance (not just URL)
    - Automatically stops when confident enough information is gathered
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
    if not openrouter_api_key and request.use_embeddings:
        print("Warning: OPENROUTER_API_KEY not set, falling back to statistical strategy", flush=True)

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
        print(f"Use embeddings: {request.use_embeddings and bool(openrouter_api_key)}", flush=True)
        print(f"{'='*60}", flush=True)

        if ADAPTIVE_AVAILABLE:
            return await run_adaptive_crawl(
                request=request,
                deepseek_api_key=deepseek_api_key,
                openrouter_api_key=openrouter_api_key,
                domain=domain
            )
        elif DEEP_CRAWL_AVAILABLE:
            return await run_fallback_deep_crawl(
                request=request,
                deepseek_api_key=deepseek_api_key,
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
    openrouter_api_key: Optional[str],
    domain: str
) -> CrawlResponse:
    """Run crawl using AdaptiveCrawler with optional embeddings."""

    # Configure adaptive strategy
    use_embedding_strategy = request.use_embeddings and openrouter_api_key

    if use_embedding_strategy:
        # Use embedding strategy with OpenRouter
        # LiteLLM format: openrouter/<model> - no base_url needed
        print(f"Using EMBEDDING strategy ({request.embedding_model})", flush=True)
        config = AdaptiveConfig(
            strategy="embedding",
            embedding_llm_config=LLMConfig(
                provider=request.embedding_model,
                api_token=openrouter_api_key
            ),
            confidence_threshold=request.confidence_threshold,
            max_pages=request.max_pages,
            top_k_links=5,
            min_gain_threshold=0.05
        )
    else:
        # Use statistical strategy (BM25) - no embeddings
        print("Using STATISTICAL (BM25) strategy", flush=True)
        config = AdaptiveConfig(
            strategy="statistical",
            confidence_threshold=request.confidence_threshold,
            max_pages=request.max_pages,
            top_k_links=5,
            min_gain_threshold=0.05
        )

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

        # Print stats
        adaptive.print_stats()

        if pages_crawled == 0:
            return CrawlResponse(
                success=False,
                answer="No pages could be crawled from the provided URL.",
                confidence=0.0,
                pages_crawled=0,
                sources=[],
                message="Crawling failed - no pages found"
            )

        # Get most relevant content
        relevant_pages = adaptive.get_relevant_content(top_k=10)

        print(f"\nTop relevant pages:", flush=True)
        for i, page in enumerate(relevant_pages[:5], 1):
            print(f"  {i}. {page['score']:.0%} - {page['url']}", flush=True)

        # Format context for DeepSeek
        context = format_adaptive_context(relevant_pages)

        if not context.strip():
            return CrawlResponse(
                success=False,
                answer="Pages were crawled but no readable content was extracted.",
                confidence=confidence,
                pages_crawled=pages_crawled,
                sources=[],
                message="No content extracted"
            )

        print(f"\nContext length: {len(context)} chars", flush=True)
        print("Generating answer with DeepSeek-reasoner...", flush=True)

        answer = await call_deepseek(
            query=request.query,
            context=context,
            api_key=deepseek_api_key
        )

        # Prepare sources
        sources = []
        for page in relevant_pages[:10]:
            sources.append({
                "url": page.get('url', 'unknown'),
                "relevance": round(page.get('score', 0), 3)
            })

        return CrawlResponse(
            success=True,
            answer=answer,
            confidence=round(confidence, 3),
            pages_crawled=pages_crawled,
            sources=sources,
            message=f"Adaptive crawl complete: {pages_crawled} pages, {confidence:.0%} confidence"
        )


async def run_fallback_deep_crawl(
    request: CrawlRequest,
    deepseek_api_key: str,
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
                message="Crawling failed - no pages found"
            )

        # Format context (simple approach for fallback)
        context_parts = []
        total_chars = 0
        max_chars = 25000

        for i, result in enumerate(successful_pages[:10], 1):
            url = result.url if hasattr(result, 'url') else str(result)
            content = ""
            if hasattr(result, 'markdown') and result.markdown:
                content = result.markdown
            elif hasattr(result, 'text') and result.text:
                content = result.text

            if not content or len(content) < 50:
                continue

            page_text = content[:5000] if len(content) > 5000 else content
            entry = f"\n=== Page {i}: {url} ===\n{page_text}\n"

            if total_chars + len(entry) > max_chars:
                break

            context_parts.append(entry)
            total_chars += len(entry)

        context = "\n".join(context_parts)

        if not context.strip():
            return CrawlResponse(
                success=False,
                answer="Pages were crawled but no readable content was extracted.",
                confidence=0.0,
                pages_crawled=pages_crawled,
                sources=[],
                message="No content extracted"
            )

        print(f"Context length: {len(context)} chars", flush=True)
        print("Generating answer with DeepSeek-reasoner...", flush=True)

        answer = await call_deepseek(
            query=request.query,
            context=context,
            api_key=deepseek_api_key
        )

        # Prepare sources
        sources = []
        for result in successful_pages[:10]:
            url = result.url if hasattr(result, 'url') else str(result)
            sources.append({"url": url, "relevance": 0.5})

        return CrawlResponse(
            success=True,
            answer=answer,
            confidence=0.5,  # No confidence score in fallback mode
            pages_crawled=pages_crawled,
            sources=sources,
            message=f"Deep crawl complete: {pages_crawled} pages (fallback mode)"
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
