"""
Crawl4AI Adaptive Crawler with DeepSeek + Sentence-Transformers
Uses Crawl4AI v0.7.8 adaptive crawling with embedding strategy.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional, List

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Test imports at startup
try:
    from crawl4ai import AsyncWebCrawler
    print("AsyncWebCrawler imported successfully", flush=True)
except ImportError as e:
    print(f"Failed to import AsyncWebCrawler: {e}", flush=True)
    sys.exit(1)

try:
    from crawl4ai import AdaptiveCrawler, AdaptiveConfig
    ADAPTIVE_AVAILABLE = True
    print("AdaptiveCrawler and AdaptiveConfig imported successfully", flush=True)
except ImportError as e:
    print(f"AdaptiveCrawler not available: {e}", flush=True)
    ADAPTIVE_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler."""
    print("=" * 50, flush=True)
    print("Crawl4AI Adaptive Crawler Starting...", flush=True)
    print(f"Adaptive Crawling Available: {ADAPTIVE_AVAILABLE}", flush=True)
    print("=" * 50, flush=True)
    yield
    print("Shutting down...", flush=True)


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Crawl4AI Adaptive Crawler",
    description="Intelligent web crawler with DeepSeek reasoning and Sentence-Transformers embeddings",
    version="1.0.0",
    lifespan=lifespan
)


class CrawlRequest(BaseModel):
    """Request model for adaptive crawling."""
    start_url: str
    query: str
    confidence_threshold: Optional[float] = 0.75
    max_pages: Optional[int] = 15
    top_k_links: Optional[int] = 5
    min_gain_threshold: Optional[float] = 0.05
    top_k_results: Optional[int] = 5


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
    max_tokens: int = 2000
) -> str:
    """Call DeepSeek API to generate an answer."""

    system_prompt = """You are a helpful assistant that provides direct, accurate answers based on the provided web content.

Your task:
1. Analyze the crawled web content provided
2. Find the most relevant information to answer the user's query
3. Provide a clear, concise, and accurate answer in plain text
4. If the information is not found in the provided content, say so honestly
5. Cite the source URLs when relevant

Be direct and informative. Focus on accuracy."""

    user_message = f"""Query: {query}

Crawled Web Content:
{context}

Please provide a direct answer to the query based on the above content."""

    async with httpx.AsyncClient(timeout=120.0) as client:
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
                "temperature": 0.3
            }
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"DeepSeek API error: {response.text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]


def format_context_for_llm(relevant_pages: List[dict], max_chars: int = 15000) -> str:
    """Format crawled pages into context for LLM."""
    context_parts = []
    total_chars = 0

    for i, page in enumerate(relevant_pages, 1):
        url = page.get("url", "Unknown URL")
        content = page.get("content", page.get("text", page.get("markdown", "")))
        score = page.get("score", page.get("relevance_score", 0))

        page_text = content[:3000] if len(content) > 3000 else content

        entry = f"""
--- Source {i} ---
URL: {url}
Relevance: {score:.0%}
Content:
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
        "version": "1.0.0",
        "adaptive_available": ADAPTIVE_AVAILABLE
    }


@app.get("/health")
async def health():
    """Health check endpoint - responds quickly."""
    return {"status": "healthy"}


@app.post("/crawl", response_model=CrawlResponse)
async def adaptive_crawl(request: CrawlRequest):
    """Perform adaptive crawling and return a direct answer."""

    if not ADAPTIVE_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="AdaptiveCrawler not available. Check crawl4ai installation."
        )

    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY environment variable not set"
        )

    try:
        # Use embedding strategy with sentence-transformers
        # Falls back to statistical if embedding fails
        try:
            config = AdaptiveConfig(
                confidence_threshold=request.confidence_threshold,
                max_pages=request.max_pages,
                top_k_links=request.top_k_links,
                min_gain_threshold=request.min_gain_threshold,
                strategy="embedding",
                embedding_model="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            print(f"Embedding strategy failed, using statistical: {e}", flush=True)
            config = AdaptiveConfig(
                confidence_threshold=request.confidence_threshold,
                max_pages=request.max_pages,
                top_k_links=request.top_k_links,
                min_gain_threshold=request.min_gain_threshold,
                strategy="statistical"
            )

        print(f"\nStarting adaptive crawl for: {request.query}", flush=True)
        print(f"Starting URL: {request.start_url}", flush=True)

        async with AsyncWebCrawler() as crawler:
            adaptive = AdaptiveCrawler(crawler, config)

            result = await adaptive.digest(
                start_url=request.start_url,
                query=request.query
            )

            adaptive.print_stats()
            relevant_pages = adaptive.get_relevant_content(top_k=request.top_k_results)

            if not relevant_pages:
                return CrawlResponse(
                    success=False,
                    answer="No relevant content could be found for your query.",
                    confidence=0.0,
                    pages_crawled=0,
                    sources=[],
                    message="Crawling completed but no relevant pages found"
                )

            context = format_context_for_llm(relevant_pages)

            print("\nGenerating answer with DeepSeek-reasoner...", flush=True)
            answer = await call_deepseek(
                query=request.query,
                context=context,
                api_key=deepseek_api_key
            )

            sources = [
                {
                    "url": page.get("url", ""),
                    "relevance_score": round(page.get("score", page.get("relevance_score", 0)), 3)
                }
                for page in relevant_pages
            ]

            confidence = getattr(adaptive, 'confidence', 0.0)
            pages_crawled = len(getattr(result, 'crawled_urls', [])) if hasattr(result, 'crawled_urls') else len(relevant_pages)

            return CrawlResponse(
                success=True,
                answer=answer,
                confidence=round(confidence, 3),
                pages_crawled=pages_crawled,
                sources=sources,
                message=f"Successfully crawled {pages_crawled} pages"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during crawl: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
