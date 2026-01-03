from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import os
import httpx
import traceback
import re
import asyncio
import json

app = FastAPI(title="Crawl4AI AI-Guided Crawler")
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

async def get_ai_keywords(query: str) -> dict:
    """Ask AI to generate relevant and irrelevant keywords for the query"""
    
    if not DEEPSEEK_API_KEY or not query:
        return {"relevant": [], "irrelevant": []}
    
    prompt = f"""Given this search query: "{query}"

Generate two lists of URL path keywords:

1. RELEVANT: Keywords that would likely appear in URLs containing useful information for this query
2. IRRELEVANT: Keywords that would likely appear in URLs NOT useful for this query

Return ONLY valid JSON, no explanation:
{{"relevant": ["keyword1", "keyword2", ...], "irrelevant": ["keyword1", "keyword2", ...]}}

Keep each list to 10-15 single-word keywords, lowercase only."""

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-reasoner",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that returns only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 300
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                content = content.strip()
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(lines[1:-1])
                
                keywords = json.loads(content)
                return keywords
            
    except Exception as e:
        print(f"AI keywords error: {e}")
    
    # Fallback defaults
    return {
        "relevant": ["docs", "guide", "tutorial", "api", "reference", "example", "faq"],
        "irrelevant": ["login", "signup", "cart", "checkout", "account", "privacy", "terms", "cookie"]
    }

def extract_relevant_content(content: str, query: str, context_lines: int = 2) -> str:
    """Extract only paragraphs/sections relevant to the query"""
    
    if not query or not content:
        return content[:1000]
    
    query_terms = []
    for term in query.split():
        term_clean = term.strip('?.,!').lower()
        if len(term_clean) > 2:
            query_terms.append(term_clean)
    
    punycode_pattern = re.findall(r'xn--[a-z0-9]+', query.lower())
    query_terms.extend(punycode_pattern)
    
    lines = content.split('\n')
    
    relevant_sections = []
    matched_indices = set()
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        for term in query_terms:
            if term in line_lower:
                matched_indices.add(i)
                break
    
    for idx in sorted(matched_indices):
        start = max(0, idx - context_lines)
        end = min(len(lines), idx + context_lines + 1)
        
        section_lines = lines[start:end]
        section = '\n'.join(line for line in section_lines if line.strip())
        
        if section and section not in relevant_sections:
            relevant_sections.append(section)
    
    if relevant_sections:
        return '\n\n---\n\n'.join(relevant_sections)
    else:
        return "[No content matching query terms found on this page]"

def score_url_relevance(url: str, link_text: str, query: str, ai_keywords: dict = None) -> float:
    """Score a URL based on how relevant it might be to the query"""
    if not query:
        return 0.5
    
    query_lower = query.lower()
    url_lower = url.lower()
    link_text_lower = link_text.lower() if link_text else ""
    
    score = 0.0
    query_terms = [t for t in query_lower.split() if len(t) > 3]
    
    # Direct query term matches
    for term in query_terms:
        if term in url_lower:
            score += 2.0
        if term in link_text_lower:
            score += 1.5
    
    # AI-generated keywords
    if ai_keywords:
        for keyword in ai_keywords.get("relevant", []):
            if keyword in url_lower or keyword in link_text_lower:
                score += 1.0
        
        for keyword in ai_keywords.get("irrelevant", []):
            if keyword in url_lower:
                score -= 1.5
    
    return max(0, score)

def get_base_domain(url: str) -> str:
    """Extract base domain from URL"""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

async def ask_deepseek(query: str, context: str) -> str:
    """Send crawled content to DeepSeek for answer"""
    
    if not DEEPSEEK_API_KEY:
        return "Error: DEEPSEEK_API_KEY not configured"
    
    context = context[:50000]
    
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
                    "max_tokens": 2000
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
    return {"message": "Crawl4AI AI-Guided Crawler", "deepseek_configured": bool(DEEPSEEK_API_KEY)}

@app.post("/crawl")
async def crawl(request: CrawlRequest, credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
        
        config = request.adaptive_crawler_config or {}
        query = config.get("query", "")
        max_pages = config.get("max_pages", 20)
        use_ai = config.get("use_ai", True)
        stay_on_domain = config.get("stay_on_domain", True)
        delay_between_requests = config.get("delay", 0.5)
        
        # 🧠 Get AI-generated keywords for this specific query
        ai_keywords = await get_ai_keywords(query) if query else {}
        
        allowed_domains = [get_base_domain(url) for url in request.urls]
        
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        content_filter = PruningContentFilter(
            threshold=0.4,
            threshold_type="fixed",
            min_word_threshold=20
        )
        
        markdown_generator = DefaultMarkdownGenerator(
            content_filter=content_filter,
            options={
                "ignore_links": False,
                "ignore_images": True,
                "body_width": 0
            }
        )
        
        crawl_config = CrawlerRunConfig(
            markdown_generator=markdown_generator,
            excluded_tags=["nav", "header", "footer", "aside", "script", "style", "noscript", "iframe", "form"],
            exclude_external_links=stay_on_domain,
            exclude_social_media_links=True,
            remove_overlay_elements=True,
            process_iframes=False
        )
        
        crawled_pages = []
        visited_urls = set()
        urls_to_crawl = [(0, url, "") for url in request.urls]
        all_content = []
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            while urls_to_crawl and len(crawled_pages) < max_pages:
                urls_to_crawl.sort(key=lambda x: x[0], reverse=True)
                priority_score, current_url, _ = urls_to_crawl.pop(0)
                
                if current_url in visited_urls or not current_url:
                    continue
                
                if stay_on_domain:
                    url_domain = get_base_domain(current_url)
                    if url_domain not in allowed_domains:
                        continue
                    
                visited_urls.add(current_url)
                
                if delay_between_requests > 0 and len(crawled_pages) > 0:
                    await asyncio.sleep(delay_between_requests)
                
                try:
                    result = await crawler.arun(url=current_url, config=crawl_config)
                    
                    if result.success:
                        markdown_content = getattr(result, 'markdown_v2', None)
                        if markdown_content and hasattr(markdown_content, 'fit_markdown'):
                            clean_content = markdown_content.fit_markdown or ""
                        else:
                            clean_content = result.markdown or ""
                        
                        lines = clean_content.split('\n')
                        clean_lines = [line.strip() for line in lines if line.strip()]
                        clean_content = '\n'.join(clean_lines)
                        
                        relevant_content = extract_relevant_content(clean_content, query)
                        
                        score = 0
                        if query:
                            query_lower = query.lower()
                            content_lower = clean_content.lower()
                            query_terms = query_lower.split()
                            matches = sum(1 for term in query_terms if term in content_lower)
                            score = matches / len(query_terms) if query_terms else 0
                        
                        title = ""
                        if hasattr(result, 'metadata') and result.metadata:
                            title = result.metadata.get("title", "")
                        
                        if clean_content:
                            all_content.append(f"## Source: {current_url}\n\n{clean_content}")
                        
                        links_found = 0
                        if hasattr(result, 'links') and result.links:
                            internal = result.links.get("internal", [])
                            links_found = len(internal)
                            
                            for link in internal[:20]:
                                if isinstance(link, dict):
                                    link_url = link.get("href", "")
                                    link_text = link.get("text", "")
                                else:
                                    link_url = str(link)
                                    link_text = ""
                                
                                if link_url and link_url not in visited_urls:
                                    already_queued = any(u[1] == link_url for u in urls_to_crawl)
                                    if not already_queued:
                                        # 🧠 Use AI keywords for scoring!
                                        url_score = score_url_relevance(link_url, link_text, query, ai_keywords)
                                        urls_to_crawl.append((url_score, link_url, link_text))
                        
                        has_relevant = "[No content matching" not in relevant_content
                        
                        crawled_pages.append({
                            "url": current_url,
                            "title": title,
                            "relevance_score": round(score, 2),
                            "has_matching_content": has_relevant,
                            "relevant_content": relevant_content,
                            "full_page_length": len(clean_content),
                            "links_found": links_found
                        })
                    else:
                        crawled_pages.append({
                            "url": current_url,
                            "relevance_score": 0,
                            "error": f"Failed: {result.error_message if hasattr(result, 'error_message') else 'Unknown'}"
                        })
                        
                except Exception as e:
                    crawled_pages.append({
                        "url": current_url,
                        "relevance_score": 0,
                        "error": str(e)
                    })
        
        ai_answer = ""
        if use_ai and query and all_content:
            combined = "\n\n---\n\n".join(all_content)
            ai_answer = await ask_deepseek(query, combined)
        
        pages_with_matches = [p for p in crawled_pages if p.get("has_matching_content", False)]
        
        return {
            "success": True,
            "query": query,
            "ai_keywords_used": ai_keywords,  # 👈 Show what AI decided!
            "answer": ai_answer,
            "pages_crawled": len(crawled_pages),
            "pages_with_matches": len(pages_with_matches),
            "domains_crawled": list(allowed_domains),
            "sources": crawled_pages
        }
        
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Import error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
