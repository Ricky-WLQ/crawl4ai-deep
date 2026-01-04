"""
Crawl4AI Adaptive Crawler with DeepSeek + Sentence-Transformers
Uses Crawl4AI v0.7.8 adaptive crawling with embedding strategy.
Intelligently crawls websites, finds relevant information, and provides direct answers.
"""

import os
from typing import Optional, List

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler, AdaptiveCrawler, AdaptiveConfig, LLMConfig

# Initialize FastAPI app
app = FastAPI(
    title="Crawl4AI Adaptive Crawler",
    description="Intelligent web crawler with DeepSeek reasoning and Sentence-Transformers embeddings",
    version="1.0.0"
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
    answer: str  # Direct plain text answer from DeepSeek
    confidence: float
    pages_crawled: int
    sources: List[dict]  # URLs and relevance scores
    message: str


async def call_deepseek(
    query: str,
    context: str,
    api_key: str,
    max_tokens: int = 2000
) -> str:
    """
    Call DeepSeek API to generate an answer based on crawled content.
    Uses deepseek-reasoner for advanced reasoning.
    """

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
            error_text = response.text
            raise HTTPException(
                status_code=response.status_code,
                detail=f"DeepSeek API error: {error_text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]


def format_context_for_llm(relevant_pages: List[dict], max_chars: int = 15000) -> str:
    """Format crawled pages into context for LLM."""
    context_parts = []
    total_chars = 0

    for i, page in enumerate(relevant_pages, 1):
        url = page.get("url", "Unknown URL")
        content = page.get("content", page.get("text", ""))
        score = page.get("score", page.get("relevance_score", 0))

        # Truncate individual page text if needed
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


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("Initializing Crawl4AI Adaptive Crawler...")
    print("Using Crawl4AI v0.7.8 with embedding strategy (all-MiniLM-L6-v2)")
    print("Service ready!")


@app.get("/")
async def root():
    """Service status endpoint."""
    return {
        "message": "Crawl4AI Adaptive Crawler is running!",
        "version": "1.0.0",
        "features": [
            "Crawl4AI v0.7.8 adaptive crawling",
            "Sentence-Transformers embeddings (all-MiniLM-L6-v2)",
            "DeepSeek-reasoner for answer generation",
            "Intelligent link prioritization"
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

    The crawler will:
    1. Start from the provided URL
    2. Use Sentence-Transformers (all-MiniLM-L6-v2) embeddings to find relevant pages
    3. Prioritize links based on relevance to the query
    4. Stop when confidence threshold is reached or max_pages hit
    5. Use DeepSeek-reasoner to generate a direct answer from the content
    """

    # Get DeepSeek API key from environment
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY environment variable not set"
        )

    try:
        # Configure adaptive crawling with embedding strategy
        # Uses Sentence-Transformers all-MiniLM-L6-v2 by default
        config = AdaptiveConfig(
            confidence_threshold=request.confidence_threshold,
            max_pages=request.max_pages,
            top_k_links=request.top_k_links,
            min_gain_threshold=request.min_gain_threshold,
            strategy="embedding",  # Use embedding-based strategy
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )

        print(f"\nStarting adaptive crawl for: {request.query}")
        print(f"Starting URL: {request.start_url}")
        print(f"Strategy: embedding (all-MiniLM-L6-v2)")

        async with AsyncWebCrawler() as crawler:
            # Initialize adaptive crawler with config
            adaptive = AdaptiveCrawler(crawler, config)

            # Start adaptive crawling
            result = await adaptive.digest(
                start_url=request.start_url,
                query=request.query
            )

            # Print crawl statistics
            adaptive.print_stats()

            # Get relevant content
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

            # Format context for DeepSeek
            context = format_context_for_llm(relevant_pages)

            # Generate answer using DeepSeek
            print("\nGenerating answer with DeepSeek-reasoner...")
            answer = await call_deepseek(
                query=request.query,
                context=context,
                api_key=deepseek_api_key
            )

            # Prepare sources
            sources = [
                {
                    "url": page.get("url", ""),
                    "relevance_score": round(page.get("score", page.get("relevance_score", 0)), 3)
                }
                for page in relevant_pages
            ]

            # Get confidence from adaptive crawler
            confidence = getattr(adaptive, 'confidence', 0.0)
            pages_crawled = len(getattr(result, 'crawled_urls', [])) if hasattr(result, 'crawled_urls') else len(relevant_pages)

            return CrawlResponse(
                success=True,
                answer=answer,
                confidence=round(confidence, 3),
                pages_crawled=pages_crawled,
                sources=sources,
                message=f"Successfully crawled and analyzed content, confidence: {confidence:.0%}"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/crawl/simple")
async def simple_crawl(start_url: str, query: str):
    """
    Simple crawl endpoint with minimal parameters.
    Uses default settings for quick crawling.
    """
    request = CrawlRequest(
        start_url=start_url,
        query=query,
        max_pages=10
    )
    return await adaptive_crawl(request)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
