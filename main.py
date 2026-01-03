"""
Crawl4AI Adaptive Crawler with DeepSeek Integration

Uses the official AdaptiveCrawler API for intelligent web crawling
with embedding-based relevance scoring and DeepSeek for answer generation.
"""
import asyncio
import os
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

# Crawl4AI imports - using official AdaptiveCrawler API
from crawl4ai import AsyncWebCrawler, AdaptiveCrawler, AdaptiveConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crawl4AI Adaptive Crawler",
    description="Intelligent adaptive web crawler with DeepSeek-powered answers",
    version="2.0.0"
)

# =============================================================================
# DeepSeek Client
# =============================================================================

def get_deepseek_client() -> OpenAI:
    """Initialize DeepSeek client using OpenAI-compatible API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )


# =============================================================================
# API Models
# =============================================================================

class CrawlRequest(BaseModel):
    """Request model for adaptive crawling."""
    start_url: str = Field(..., description="The starting URL to crawl")
    query: str = Field(..., description="The question or topic to search for")
    max_pages: int = Field(default=20, ge=1, le=50, description="Maximum pages to crawl (1-50)")
    confidence_threshold: float = Field(default=0.7, ge=0.1, le=1.0, description="Stop when confidence reaches this level (0.1-1.0)")
    top_k_links: int = Field(default=3, ge=1, le=10, description="Number of top links to follow per page (1-10)")
    min_gain_threshold: float = Field(default=0.1, ge=0.0, le=0.5, description="Minimum information gain to continue crawling")
    strategy: str = Field(default="statistical", description="Crawling strategy: 'statistical' or 'embedding'")


class PageContent(BaseModel):
    """Model for crawled page content."""
    url: str
    score: float
    content_preview: str


class CrawlResponse(BaseModel):
    """Response model for crawl results."""
    success: bool
    answer: str
    confidence: float
    pages_crawled: int
    relevant_pages: List[PageContent]
    query: str
    message: str
    strategy: str


# =============================================================================
# Helper Functions
# =============================================================================

async def process_with_deepseek(
    query: str,
    relevant_pages: List[Dict[str, Any]],
    client: OpenAI
) -> str:
    """Process crawled content with DeepSeek to generate an answer."""
    
    # Prepare context from relevant pages
    context_parts = []
    total_chars = 0
    max_chars = 25000
    
    for page in relevant_pages:
        if total_chars >= max_chars:
            break
        content = page.get('content', '')[:4000] if page.get('content') else ''
        url = page.get('url', '')
        score = page.get('score', 0)
        context_parts.append(f"[Source: {url} | Relevance: {score:.2f}]\n{content}\n")
        total_chars += len(content)
    
    if not context_parts:
        return "No relevant content was found to answer your question."
    
    context = "\n---\n".join(context_parts)
    
    system_prompt = """You are an intelligent assistant that analyzes web content to answer questions accurately.

Your task:
1. Carefully read the provided web content from multiple pages (sorted by relevance)
2. Extract the most relevant information to answer the user's question
3. Synthesize the information into a clear, direct answer
4. If the information is not found in the content, clearly state that
5. Cite sources when providing specific facts

Be concise but comprehensive. Provide a direct answer in plain text."""

    user_prompt = f"""Based on the following web content (sorted by relevance to your query), please answer this question:

Question: {query}

Web Content:
{context}

Please provide a clear, accurate answer based on the information above."""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"DeepSeek API error: {e}")
        return f"Error processing with DeepSeek: {str(e)}"


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint - service status."""
    return {
        "service": "Crawl4AI Adaptive Crawler",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "Official AdaptiveCrawler API",
            "Statistical and Embedding strategies",
            "Confidence-based stopping",
            "Intelligent link selection",
            "DeepSeek AI-powered answer generation"
        ],
        "endpoints": {
            "crawl": "POST /crawl - Perform adaptive crawl with AI answer",
            "health": "GET /health - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    deepseek_configured = bool(os.environ.get("DEEPSEEK_API_KEY"))
    return {
        "status": "healthy",
        "deepseek_configured": deepseek_configured
    }


@app.post("/crawl", response_model=CrawlResponse)
async def adaptive_crawl(request: CrawlRequest):
    """
    Perform adaptive crawling using Crawl4AI's official AdaptiveCrawler.
    
    The crawler will:
    1. Start from the given URL
    2. Intelligently follow relevant links based on the query
    3. Stop when confidence threshold is reached or max_pages hit
    4. Use DeepSeek to analyze content and provide a direct answer
    
    Strategies:
    - "statistical": Fast, keyword-based relevance scoring
    - "embedding": Semantic understanding using embeddings (slower but more accurate)
    """
    try:
        # Configure adaptive crawling
        config = AdaptiveConfig(
            strategy=request.strategy,
            confidence_threshold=request.confidence_threshold,
            max_pages=request.max_pages,
            top_k_links=request.top_k_links,
            min_gain_threshold=request.min_gain_threshold
        )
        
        logger.info(f"Starting adaptive crawl: {request.start_url}")
        logger.info(f"Query: {request.query}")
        logger.info(f"Strategy: {request.strategy}, Max pages: {request.max_pages}")
        
        async with AsyncWebCrawler() as crawler:
            # Create adaptive crawler
            adaptive = AdaptiveCrawler(crawler, config)
            
            # Perform adaptive crawling
            result = await adaptive.digest(
                start_url=request.start_url,
                query=request.query
            )
            
            # Get confidence and stats
            confidence = adaptive.confidence
            crawled_urls = result.crawled_urls if hasattr(result, 'crawled_urls') else []
            
            # Get relevant content
            relevant_content = adaptive.get_relevant_content(top_k=10)
            
            logger.info(f"Crawled {len(crawled_urls)} pages, confidence: {confidence:.0%}")
        
        if not relevant_content:
            return CrawlResponse(
                success=False,
                answer="No relevant content could be found for your query.",
                confidence=confidence,
                pages_crawled=len(crawled_urls),
                relevant_pages=[],
                query=request.query,
                message="Crawling completed but no relevant content found",
                strategy=request.strategy
            )
        
        # Process with DeepSeek
        deepseek_client = get_deepseek_client()
        answer = await process_with_deepseek(
            request.query,
            relevant_content,
            deepseek_client
        )
        
        # Prepare response
        relevant_pages = [
            PageContent(
                url=page.get('url', ''),
                score=round(page.get('score', 0), 4),
                content_preview=page.get('content', '')[:300] + "..." if page.get('content') and len(page.get('content', '')) > 300 else page.get('content', '')
            )
            for page in relevant_content[:10]
        ]
        
        return CrawlResponse(
            success=True,
            answer=answer,
            confidence=round(confidence, 4),
            pages_crawled=len(crawled_urls),
            relevant_pages=relevant_pages,
            query=request.query,
            message=f"Successfully crawled {len(crawled_urls)} pages with {confidence:.0%} confidence using {request.strategy} strategy",
            strategy=request.strategy
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Crawling error")
        raise HTTPException(status_code=500, detail=f"Crawling error: {str(e)}")


@app.post("/crawl/simple")
async def simple_crawl(url: str, query: str):
    """
    Simple single-page crawl with DeepSeek analysis.
    Uses default AdaptiveCrawler settings for quick results.
    """
    try:
        async with AsyncWebCrawler() as crawler:
            adaptive = AdaptiveCrawler(crawler)
            
            result = await adaptive.digest(
                start_url=url,
                query=query
            )
            
            relevant_content = adaptive.get_relevant_content(top_k=5)
            
            if not relevant_content:
                return {
                    "success": False,
                    "url": url,
                    "answer": "No relevant content found.",
                    "confidence": adaptive.confidence
                }
            
            # Process with DeepSeek
            deepseek_client = get_deepseek_client()
            answer = await process_with_deepseek(
                query,
                relevant_content,
                deepseek_client
            )
            
            return {
                "success": True,
                "url": url,
                "answer": answer,
                "confidence": round(adaptive.confidence, 4),
                "pages_crawled": len(result.crawled_urls) if hasattr(result, 'crawled_urls') else 0
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
