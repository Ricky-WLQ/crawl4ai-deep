from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
import os
import httpx

app = FastAPI(title="Crawl4AI Adaptive Crawler")
security = HTTPBearer()

PASSWORD = os.getenv("PASSWORD", "changeme")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

class CrawlRequest(BaseModel):
    urls: list[str]
    mode: str = "adaptive"
    adaptive_crawler_config: dict = None

async def ask_deepseek(query: str, context: str) -> str:
    """Send crawled content to DeepSeek-Reasoner and get plain language answer"""
    
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API key not configured"
    
    # Limit context size to avoid token limits
    context = context[:15000]
    
    prompt = f"""Based on the following web content, please answer this question in plain language:

**Question:** {query}

**Web Content:**
{context}

**Instructions:**
- Answer directly and concisely
- Use bullet points for lists
- If the information isn't found, say so
- Cite the source URL when relevant
"""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-reasoner",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful research assistant. Analyze web content and provide clear, accurate answers."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"DeepSeek API error: {response.status_code} - {response.text}"
                
    except Exception as e:
        return f"Error calling DeepSeek: {str(e)}"

@app.get("/")
async def root():
    return {"message": "Crawl4AI Adaptive Crawler with DeepSeek-Reasoner is running!"}

@app.post("/crawl")
async def crawl(request: CrawlRequest, credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    try:
        config = request.adaptive_crawler_config or {}
        query = config.get("query", "")
        max_pages = config.get("max_pages", 10)
        use_ai = config.get("use_ai", True)  # Enable/disable AI answer
        
        browser_config = BrowserConfig(headless=True)
        crawl_config = CrawlerRunConfig()
        
        crawled_pages = []
        visited_urls = set()
        urls_to_crawl = list(request.urls)
        all_content = []  # Collect content for DeepSeek
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            while urls_to_crawl and len(crawled_pages) < max_pages:
                current_url = urls_to_crawl.pop(0)
                
                if current_url in visited_urls:
                    continue
                    
                visited_urls.add(current_url)
                
                try:
                    result = await crawler.arun(url=current_url, config=crawl_config)
                    
                    if result.success:
                        # Calculate relevance score
                        score = 0
                        markdown_content = result.markdown or ""
                        
                        if query.lower() in markdown_content.lower():
                            score = 1.0
                        elif markdown_content:
                            score = min(len(markdown_content) / 5000, 0.75)
                        
                        # Get page title
                        title = ""
                        if result.metadata and "title" in result.metadata:
                            title = result.metadata["title"]
                        
                        # Collect content for AI processing
                        if markdown_content:
                            all_content.append(f"## Source: {current_url}\n### {title}\n{markdown_content[:3000]}")
                        
                        # Get internal links
                        internal_links = []
                        if result.links:
                            for link in result.links.get("internal", []):
                                link_url = link.get("href", "") if isinstance(link, dict) else link
                                if link_url and link_url not in visited_urls:
                                    internal_links.append(link_url)
                                    if link_url not in urls_to_crawl:
                                        urls_to_crawl.append(link_url)
                        
                        crawled_pages.append({
                            "url": current_url,
                            "score": round(score, 2),
                            "title": title,
                            "content_preview": markdown_content[:500] if markdown_content else "",
                            "links_found": len(internal_links)
                        })
                        
                except Exception as e:
                    crawled_pages.append({
                        "url": current_url,
                        "score": 0,
                        "title": "",
                        "content_preview": "",
                        "links_found": 0,
                        "error": str(e)
                    })
        
        # Combine all content for DeepSeek
        combined_content = "\n\n---\n\n".join(all_content)
        
        # Get AI answer if enabled
        ai_answer = ""
        if use_ai and query and combined_content:
            ai_answer = await ask_deepseek(query, combined_content)
        
        # Calculate average confidence
        total_score = sum(p.get("score", 0) for p in crawled_pages)
        avg_confidence = total_score / len(crawled_pages) if crawled_pages else 0
        
        return {
            "success": True,
            "query": query,
            "answer": ai_answer,  # 👈 PLAIN LANGUAGE ANSWER!
            "confidence": round(avg_confidence, 2),
            "pages_crawled": len(crawled_pages),
            "sources": crawled_pages,
            "message": f"Crawled {len(crawled_pages)} pages"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
