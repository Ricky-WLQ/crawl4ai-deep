import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from crawl4ai import AsyncWebCrawler, AdaptiveCrawler, AdaptiveConfig

app = FastAPI(title="Crawl4AI Adaptive Crawler")

class CrawlRequest(BaseModel):
    start_url: str
    query: str
    confidence_threshold: Optional[float] = 0.7
    max_pages: Optional[int] = 20
    top_k_links: Optional[int] = 3
    min_gain_threshold: Optional[float] = 0.1

class CrawlResponse(BaseModel):
    success: bool
    confidence: float
    pages_crawled: int
    relevant_content: List[dict]
    message: str

@app.get("/")
async def root():
    return {"message": "Crawl4AI Adaptive Crawler is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/crawl", response_model=CrawlResponse)
async def adaptive_crawl(request: CrawlRequest):
    try:
        # Configure adaptive crawling
        config = AdaptiveConfig(
            confidence_threshold=request.confidence_threshold,
            max_pages=request.max_pages,
            top_k_links=request.top_k_links,
            min_gain_threshold=request.min_gain_threshold
        )
        
        async with AsyncWebCrawler() as crawler:
            adaptive = AdaptiveCrawler(crawler, config)
            
            # Start adaptive crawling
            result = await adaptive.digest(
                start_url=request.start_url,
                query=request.query
            )
            
            # Get relevant content
            relevant_pages = adaptive.get_relevant_content(top_k=5)
            
            return CrawlResponse(
                success=True,
                confidence=adaptive.confidence,
                pages_crawled=len(result.crawled_urls),
                relevant_content=[
                    {"url": page["url"], "score": page["score"]} 
                    for page in relevant_pages
                ],
                message=f"Achieved {adaptive.confidence:.0%} confidence"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)