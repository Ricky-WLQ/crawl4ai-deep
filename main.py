"""
Crawl4AI Adaptive Crawler with Embedding-Based Relevance Scoring and DeepSeek Integration

Features:
- Embedding-based semantic relevance scoring (multilingual support)
- Adaptive crawling that intelligently discovers relevant subpages
- DeepSeek integration for AI-powered answer generation
"""
import asyncio
import os
import json
import logging
from typing import Optional, List, Dict, Any, Union
from urllib.parse import urlparse, urljoin
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import numpy as np

# Crawl4AI imports
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, DomainFilter, ContentTypeFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crawl4AI Adaptive Crawler with Embeddings",
    description="Intelligent semantic web crawler with embedding-based relevance scoring and DeepSeek-powered answers",
    version="2.0.0"
)

# =============================================================================
# Embedding Model (Multilingual)
# =============================================================================

class EmbeddingModel:
    """
    Multilingual embedding model using sentence-transformers.
    Uses paraphrase-multilingual-mpnet-base-v2 for 50+ language support.
    """
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if EmbeddingModel._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the multilingual embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading multilingual embedding model...")
            # paraphrase-multilingual-mpnet-base-v2: 50+ languages, 768 dimensions
            EmbeddingModel._model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
        
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = EmbeddingModel._model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return embeddings
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        If embeddings are normalized, this is just the dot product.
        """
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        return float(np.dot(embedding1, embedding2.T)[0, 0])
    
    def batch_similarity(self, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Calculate similarity between a query and multiple embeddings."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        return np.dot(embeddings, query_embedding.T).flatten()


# Global embedding model instance (lazy loaded)
_embedding_model: Optional[EmbeddingModel] = None

def get_embedding_model() -> EmbeddingModel:
    """Get or create the embedding model singleton."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


# =============================================================================
# Custom Embedding-Based Scorer for Crawl4AI
# =============================================================================

class EmbeddingRelevanceScorer:
    """
    Custom scorer that uses embeddings for semantic URL/link relevance scoring.
    Compatible with Crawl4AI's BestFirstCrawlingStrategy.
    """
    
    def __init__(self, query: str, weight: float = 1.0):
        """
        Initialize the embedding scorer.
        
        Args:
            query: The search query to score relevance against
            weight: Weight multiplier for the score (0.0 to 1.0)
        """
        self.query = query
        self.weight = weight
        self.embedding_model = get_embedding_model()
        self.query_embedding = self.embedding_model.encode(query)
        
        # Cache for computed embeddings
        self._cache: Dict[str, float] = {}
    
    def score(self, url: str, link_text: str = "", context: str = "") -> float:
        """
        Score a URL based on semantic similarity to the query.
        
        Args:
            url: The URL to score
            link_text: The anchor text of the link
            context: Additional context (surrounding text)
        
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Create cache key
        cache_key = f"{url}:{link_text}:{context[:100]}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Combine URL path, link text, and context for scoring
        # Extract meaningful parts from URL
        parsed = urlparse(url)
        url_text = parsed.path.replace('/', ' ').replace('-', ' ').replace('_', ' ')
        
        # Combine all text signals
        combined_text = f"{link_text} {url_text} {context}".strip()
        
        if not combined_text:
            return 0.0
        
        # Compute embedding and similarity
        text_embedding = self.embedding_model.encode(combined_text)
        similarity = self.embedding_model.similarity(self.query_embedding, text_embedding)
        
        # Normalize to 0-1 range (cosine similarity is already -1 to 1)
        score = max(0.0, (similarity + 1) / 2) * self.weight
        
        # Cache the result
        self._cache[cache_key] = score
        
        return score
    
    def __call__(self, url: str, **kwargs) -> float:
        """Make the scorer callable for Crawl4AI compatibility."""
        link_text = kwargs.get('link_text', kwargs.get('text', ''))
        context = kwargs.get('context', '')
        return self.score(url, link_text, context)


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
    max_depth: int = Field(default=2, ge=1, le=5, description="Maximum crawl depth (1-5)")
    max_pages: int = Field(default=15, ge=1, le=50, description="Maximum pages to crawl (1-50)")
    include_external: bool = Field(default=False, description="Whether to follow external links")
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum relevance score for URLs")
    use_keywords_fallback: bool = Field(default=False, description="Use keyword scorer as fallback if embeddings fail")


class PageContent(BaseModel):
    """Model for crawled page content."""
    url: str
    title: Optional[str] = None
    relevance_score: float
    content_preview: str
    depth: int = 0


class CrawlResponse(BaseModel):
    """Response model for crawl results."""
    success: bool
    answer: str
    pages_crawled: int
    relevant_pages: List[PageContent]
    query: str
    message: str
    scoring_method: str = "embedding"


# =============================================================================
# Helper Functions
# =============================================================================

def get_domain_from_url(url: str) -> str:
    """Extract domain from URL."""
    parsed = urlparse(url)
    return parsed.netloc


def extract_keywords(query: str) -> List[str]:
    """Extract keywords from query for fallback keyword scoring."""
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'what', 'which', 'who',
        'whom', 'this', 'that', 'these', 'those', 'and', 'but', 'if', 'or',
        'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        'about', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
        'under', 'again', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
        'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'just', 'now', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'it'
    }
    
    words = re.sub(r'[^\w\s]', '', query.lower()).split()
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    if len(keywords) < 2:
        keywords = [word for word in words if len(word) > 2]
    
    return keywords[:10]


def score_content_relevance(
    query: str,
    contents: List[Dict[str, Any]],
    embedding_model: EmbeddingModel
) -> List[Dict[str, Any]]:
    """
    Score crawled content by semantic relevance to the query using embeddings.
    
    Args:
        query: The search query
        contents: List of crawled page data with 'content' field
        embedding_model: The embedding model to use
    
    Returns:
        List of contents with 'relevance_score' added, sorted by relevance
    """
    if not contents:
        return []
    
    # Encode query
    query_embedding = embedding_model.encode(query)
    
    # Encode all content (use first 2000 chars for efficiency)
    content_texts = []
    for page in contents:
        content = page.get('content', '')[:2000]
        title = page.get('title', '')
        url_text = urlparse(page.get('url', '')).path.replace('/', ' ')
        combined = f"{title} {url_text} {content}".strip()
        content_texts.append(combined if combined else "empty")
    
    # Batch encode content
    content_embeddings = embedding_model.encode(content_texts)
    
    # Calculate similarities
    similarities = embedding_model.batch_similarity(query_embedding, content_embeddings)
    
    # Add scores to contents
    for i, page in enumerate(contents):
        # Normalize to 0-1 range
        page['relevance_score'] = float(max(0, (similarities[i] + 1) / 2))
    
    # Sort by relevance
    contents.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return contents


async def process_with_deepseek(
    query: str,
    crawled_content: List[Dict[str, Any]],
    client: OpenAI
) -> str:
    """Process crawled content with DeepSeek to generate an answer."""
    
    # Prepare context from crawled pages (limit to avoid token overflow)
    context_parts = []
    total_chars = 0
    max_chars = 25000
    
    for page in crawled_content:
        if total_chars >= max_chars:
            break
        content = page.get('content', '')[:4000]
        url = page.get('url', '')
        score = page.get('relevance_score', 0)
        context_parts.append(f"[Source: {url} | Relevance: {score:.2f}]\n{content}\n")
        total_chars += len(content)
    
    context = "\n---\n".join(context_parts)
    
    system_prompt = """You are an intelligent assistant that analyzes web content to answer questions accurately.

Your task:
1. Carefully read the provided web content from multiple pages (sorted by relevance)
2. Extract the most relevant information to answer the user's question
3. Synthesize the information into a clear, direct answer
4. If the information is not found in the content, clearly state that
5. Cite sources when providing specific facts

Be concise but comprehensive. Provide a direct answer in plain text."""

    user_prompt = f"""Based on the following web content (sorted by semantic relevance to your query), please answer this question:

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
# Main Crawling Logic
# =============================================================================

async def perform_adaptive_crawl(request: CrawlRequest) -> Dict[str, Any]:
    """
    Perform adaptive crawling with embedding-based relevance scoring.
    
    Uses BestFirstCrawlingStrategy with either:
    - EmbeddingRelevanceScorer (primary): Semantic similarity scoring
    - KeywordRelevanceScorer (fallback): Keyword-based scoring
    """
    
    # Initialize embedding model
    try:
        embedding_model = get_embedding_model()
        use_embeddings = True
        logger.info("Using embedding-based relevance scoring")
    except Exception as e:
        logger.warning(f"Embedding model unavailable: {e}. Falling back to keywords.")
        use_embeddings = False
    
    # Get domain for filtering
    start_domain = get_domain_from_url(request.start_url)
    
    # Set up filters
    filters = [ContentTypeFilter(allowed_types=["text/html"])]
    if not request.include_external:
        filters.insert(0, DomainFilter(allowed_domains=[start_domain]))
    
    filter_chain = FilterChain(filters)
    
    # Create scorer based on available method
    scoring_method = "embedding"
    if use_embeddings and not request.use_keywords_fallback:
        try:
            url_scorer = EmbeddingRelevanceScorer(
                query=request.query,
                weight=0.9
            )
            scoring_method = "embedding"
        except Exception as e:
            logger.warning(f"Failed to create embedding scorer: {e}")
            use_embeddings = False
    
    if not use_embeddings or request.use_keywords_fallback:
        keywords = extract_keywords(request.query)
        url_scorer = KeywordRelevanceScorer(
            keywords=keywords,
            weight=0.8
        )
        scoring_method = "keyword"
        logger.info(f"Using keyword-based scoring with keywords: {keywords}")
    
    # Configure browser
    browser_config = BrowserConfig(
        headless=True,
        viewport_width=1280,
        viewport_height=720
    )
    
    # Configure crawler with BestFirstCrawlingStrategy for adaptive behavior
    crawler_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=request.max_depth,
            include_external=request.include_external,
            url_scorer=url_scorer,
            filter_chain=filter_chain,
            max_pages=request.max_pages,
            score_threshold=request.score_threshold
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        word_count_threshold=50,
        remove_overlay_elements=True,
        process_iframes=True
    )
    
    # Perform crawling
    crawled_pages = []
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        logger.info(f"Starting adaptive crawl from {request.start_url}")
        
        results = await crawler.arun(
            url=request.start_url,
            config=crawler_config
        )
        
        # Handle both single result and list of results
        if isinstance(results, list):
            results_list = results
        else:
            results_list = [results]
        
        for result in results_list:
            if result.success:
                # Extract content
                content = ""
                if hasattr(result, 'markdown') and result.markdown:
                    if hasattr(result.markdown, 'raw_markdown'):
                        content = result.markdown.raw_markdown
                    elif hasattr(result.markdown, 'fit_markdown'):
                        content = result.markdown.fit_markdown
                    else:
                        content = str(result.markdown)
                elif hasattr(result, 'cleaned_html'):
                    content = result.cleaned_html[:10000]
                
                # Get metadata
                url_score = 0.5
                depth = 0
                title = None
                
                if hasattr(result, 'metadata') and result.metadata:
                    url_score = result.metadata.get('score', 0.5)
                    depth = result.metadata.get('depth', 0)
                    title = result.metadata.get('title')
                
                crawled_pages.append({
                    'url': result.url,
                    'title': title,
                    'content': content,
                    'url_score': url_score,
                    'depth': depth
                })
    
    logger.info(f"Crawled {len(crawled_pages)} pages")
    
    # Score content relevance using embeddings (if available)
    if use_embeddings and crawled_pages:
        crawled_pages = score_content_relevance(
            request.query,
            crawled_pages,
            embedding_model
        )
    else:
        # Use URL score as relevance score for fallback
        for page in crawled_pages:
            page['relevance_score'] = page.get('url_score', 0.5)
        crawled_pages.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return {
        'pages': crawled_pages,
        'scoring_method': scoring_method
    }


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint - service status."""
    return {
        "service": "Crawl4AI Adaptive Crawler with Embeddings",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "Embedding-based semantic relevance scoring",
            "Multilingual support (50+ languages)",
            "Adaptive crawling with BestFirstCrawlingStrategy",
            "DeepSeek AI-powered answer generation"
        ],
        "endpoints": {
            "crawl": "POST /crawl - Perform adaptive crawl with AI answer",
            "crawl_simple": "POST /crawl/simple - Single page crawl",
            "health": "GET /health - Health check",
            "warmup": "POST /warmup - Pre-load embedding model"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    deepseek_configured = bool(os.environ.get("DEEPSEEK_API_KEY"))
    
    embedding_status = "not_loaded"
    try:
        if _embedding_model is not None:
            embedding_status = "loaded"
    except:
        pass
    
    return {
        "status": "healthy",
        "deepseek_configured": deepseek_configured,
        "embedding_model": embedding_status,
        "embedding_model_name": "paraphrase-multilingual-mpnet-base-v2"
    }


@app.post("/warmup")
async def warmup():
    """Pre-load the embedding model to reduce first-request latency."""
    try:
        model = get_embedding_model()
        # Test encoding
        _ = model.encode("test warmup")
        return {"status": "success", "message": "Embedding model loaded and ready"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


@app.post("/crawl", response_model=CrawlResponse)
async def adaptive_crawl(request: CrawlRequest):
    """
    Perform adaptive crawling with embedding-based relevance scoring.
    
    The crawler will:
    1. Start from the given URL
    2. Use semantic embeddings to score URL/link relevance
    3. Prioritize and follow the most relevant links (BestFirstCrawlingStrategy)
    4. Collect content from crawled pages
    5. Re-score content by semantic similarity to the query
    6. Use DeepSeek to analyze content and provide a direct answer
    """
    try:
        # Perform adaptive crawl
        crawl_result = await perform_adaptive_crawl(request)
        crawled_pages = crawl_result['pages']
        scoring_method = crawl_result['scoring_method']
        
        if not crawled_pages:
            return CrawlResponse(
                success=False,
                answer="No pages could be crawled from the provided URL.",
                pages_crawled=0,
                relevant_pages=[],
                query=request.query,
                message="Crawling failed - no accessible pages found",
                scoring_method=scoring_method
            )
        
        # Process with DeepSeek
        deepseek_client = get_deepseek_client()
        answer = await process_with_deepseek(
            request.query,
            crawled_pages,
            deepseek_client
        )
        
        # Prepare response
        relevant_pages = [
            PageContent(
                url=page['url'],
                title=page.get('title'),
                relevance_score=round(page.get('relevance_score', 0), 4),
                content_preview=page['content'][:300] + "..." if len(page['content']) > 300 else page['content'],
                depth=page.get('depth', 0)
            )
            for page in crawled_pages[:10]
        ]
        
        return CrawlResponse(
            success=True,
            answer=answer,
            pages_crawled=len(crawled_pages),
            relevant_pages=relevant_pages,
            query=request.query,
            message=f"Successfully crawled {len(crawled_pages)} pages using {scoring_method}-based scoring",
            scoring_method=scoring_method
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Crawling error")
        raise HTTPException(status_code=500, detail=f"Crawling error: {str(e)}")


@app.post("/crawl/simple")
async def simple_crawl(url: str, query: str):
    """
    Simple single-page crawl with embedding-based content scoring and DeepSeek analysis.
    """
    try:
        browser_config = BrowserConfig(headless=True)
        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=50
        )
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawler_config)
            
            if not result.success:
                raise HTTPException(status_code=400, detail="Failed to crawl the URL")
            
            content = ""
            if hasattr(result, 'markdown') and result.markdown:
                if hasattr(result.markdown, 'raw_markdown'):
                    content = result.markdown.raw_markdown
                else:
                    content = str(result.markdown)
            
            # Score content relevance
            embedding_model = get_embedding_model()
            pages = [{'url': url, 'content': content}]
            scored_pages = score_content_relevance(query, pages, embedding_model)
            relevance_score = scored_pages[0].get('relevance_score', 0)
            
            # Process with DeepSeek
            deepseek_client = get_deepseek_client()
            answer = await process_with_deepseek(
                query,
                scored_pages,
                deepseek_client
            )
            
            return {
                "success": True,
                "url": url,
                "answer": answer,
                "relevance_score": round(relevance_score, 4),
                "content_length": len(content)
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
