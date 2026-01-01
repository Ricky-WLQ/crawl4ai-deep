from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import os
import httpx
import traceback

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
    adaptive_crawler_config: Optional[dict] = None

async def ask_deepseek(query: str, context: str) -> str:
    """Send crawled content to DeepSeek-Reasoner"""
    
    if not DEEPSEEK_API_KEY:
        return "Error: DEEPSEEK_API_KEY not configured"
    
    context = context[:50000]  # Increased context limit
    
    prompt = f"""Based on the following web content, please answer this question:

**Question:** {query}

**Web Content:**
{context}

**Instructions:**
- Answer directly and concisely
- Use bullet points for lists
- If the information isn't found, say so
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
                        {"role": "system", "content": "You are a helpful research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 4000
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"DeepSeek API error: {response.status_code}"
                
    except Exception as e:
        return f"Error calling DeepSeek: {str(e)}"

@app.get("/")
async def root():
    return {"message": "Crawl4AI Adaptive Crawler is running!", "deepseek_configured": bool(DEEPSEEK_API_KEY)}

@app.post("/crawl")
async def crawl(request: CrawlRequest, credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
        
        config = request.adaptive_crawler_config or {}
        query = config.get("query", "")
        max_pages = config.get("max_pages", 5)
        use_ai = config.get("use_ai", True)
        
        # Browser config - headless mode
        browser_config = BrowserConfig(
            headless=True,
            verbose=False
        )
        
        # Content filter - removes boilerplate, keeps only main content
        content_filter = PruningContentFilter(
            threshold=0.4,  # Lower = more content kept, Higher = stricter filtering
            threshold_type="fixed",
            min_word_threshold=20  # Minimum words per block to keep
        )
        
        # Markdown generator - clean output
        markdown_generator = DefaultMarkdownGenerator(
            content_filter=content_filter,
            options={
                "ignore_links": True,  # Remove link clutter
                "ignore_images": True,  # Remove image references
                "body_width": 0  # No text wrapping
            }
        )
        
        # Crawler config with smart extraction
        crawl_config = CrawlerRunConfig(
            markdown_generator=markdown_generator,
            excluded_tags=["nav", "header", "footer", "aside", "script", "style", "noscript", "iframe", "form"],
            exclude_external_links=True,
            exclude_social_media_links=True,
            remove_overlay_elements=True,
            process_iframes=False
        )
        
        crawled_pages = []
        visited_urls = set()
        urls_to_crawl = list(request.urls)
        all_content = []
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            while urls_to_crawl and len(crawled_pages) < max_pages:
                current_url = urls_to_crawl.pop(0)
                
                if current_url in visited_urls or not current_url:
                    continue
                    
                visited_urls.add(current_url)
                
                try:
                    result = await crawler.arun(url=current_url, config=crawl_config)
                    
                    if result.success:
                        # Use fit_markdown (filtered) if available, otherwise regular markdown
                        markdown_content = getattr(result, 'markdown_v2', None)
                        if markdown_content and hasattr(markdown_content, 'fit_markdown'):
                            clean_content = markdown_content.fit_markdown or ""
                        else:
                            clean_content = result.markdown or ""
                        
                        # Additional cleanup - remove empty lines and extra whitespace
                        lines = clean_content.split('\n')
                        clean_lines = [line.strip() for line in lines if line.strip()]
                        clean_content = '\n'.join(clean_lines)
                        
                        # Score based on query match
                        score = 0
                        if query:
                            query_lower = query.lower()
                            content_lower = clean_content.lower()
                            # Check for query terms
                            query_terms = query_lower.split()
                            matches = sum(1 for term in query_terms if term in content_lower)
                            score = matches / len(query_terms) if query_terms else 0
                        
                        # Get title
                        title = ""
                        if hasattr(result, 'metadata') and result.metadata:
                            title = result.metadata.get("title", "")
                        
                        # Store FULL clean content (no truncation for AI)
                        if clean_content:
                            all_content.append(f"## Source: {current_url}\n\n{clean_content}")
                        
                        # Follow internal links
                        links_found = 0
                        if hasattr(result, 'links') and result.links:
                            internal = result.links.get("internal", [])
                            links_found = len(internal)
                            for link in internal[:10]:
                                link_url = link.get("href", "") if isinstance(link, dict) else str(link)
                                if link_url and link_url not in visited_urls and link_url not in urls_to_crawl:
                                    urls_to_crawl.append(link_url)
                        
                        crawled_pages.append({
                            "url": current_url,
                            "score": round(score, 2),
                            "title": title,
                            "content_length": len(clean_content),
                            "content_preview": clean_content[:500],  # Just for response preview
                            "links_found": links_found
                        })
                        
                except Exception as e:
                    crawled_pages.append({
                        "url": current_url,
                        "score": 0,
                        "error": str(e)
                    })
        
        # Send ALL content to AI (not truncated)
        ai_answer = ""
        if use_ai and query and all_content:
            combined = "\n\n---\n\n".join(all_content)
            ai_answer = await ask_deepseek(query, combined)
        
        total_score = sum(p.get("score", 0) for p in crawled_pages)
        avg_confidence = total_score / len(crawled_pages) if crawled_pages else 0
        
        return {
            "success": True,
            "query": query,
            "answer": ai_answer,
            "confidence": round(avg_confidence, 2),
            "pages_crawled": len(crawled_pages),
            "total_content_extracted": sum(p.get("content_length", 0) for p in crawled_pages),
            "sources": crawled_pages
        }
        
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Import error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
