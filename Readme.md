# Crawl4AI Adaptive Crawler with DeepSeek

An intelligent web crawler using Crawl4AI's official **AdaptiveCrawler** API with **DeepSeek** for AI-powered answer generation.

## 🌟 Features

- **Official AdaptiveCrawler API**: Uses crawl4ai's built-in adaptive crawling
- **Two Strategies**:
  - `statistical`: Fast keyword-based relevance (default)
  - `embedding`: Semantic understanding with embeddings
- **Confidence-based Stopping**: Automatically stops when sufficient information gathered
- **Intelligent Link Selection**: Follows only the most relevant links
- **DeepSeek Integration**: Generates direct answers from crawled content

## 🚀 Setup

### Environment Variables (Zeabur Dashboard)

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPSEEK_API_KEY` | Yes | Your DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com/) |
| `PORT` | No | Server port (default: 8080) |

### Zeabur Deployment

1. Push code to your GitHub repository
2. Connect to Zeabur
3. Add `DEEPSEEK_API_KEY` in Variables tab
4. Deploy!

## 📡 API Endpoints

### `POST /crawl`

Main adaptive crawling endpoint.

**Request Body:**
```json
{
  "start_url": "https://example.com",
  "query": "What are the pricing plans?",
  "max_pages": 20,
  "confidence_threshold": 0.7,
  "top_k_links": 3,
  "min_gain_threshold": 0.1,
  "strategy": "statistical"
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_url` | string | required | Starting URL to crawl |
| `query` | string | required | Question or topic to search for |
| `max_pages` | int | 20 | Maximum pages to crawl (1-50) |
| `confidence_threshold` | float | 0.7 | Stop when confidence reaches this (0.1-1.0) |
| `top_k_links` | int | 3 | Links to follow per page (1-10) |
| `min_gain_threshold` | float | 0.1 | Minimum info gain to continue (0.0-0.5) |
| `strategy` | string | "statistical" | "statistical" or "embedding" |

**Response:**
```json
{
  "success": true,
  "answer": "Based on the website, the pricing plans include...",
  "confidence": 0.85,
  "pages_crawled": 8,
  "relevant_pages": [
    {
      "url": "https://example.com/pricing",
      "score": 0.92,
      "content_preview": "Our pricing starts at..."
    }
  ],
  "query": "What are the pricing plans?",
  "message": "Successfully crawled 8 pages with 85% confidence",
  "strategy": "statistical"
}
```

### `GET /health`

Health check endpoint.

### `GET /`

Service info and available endpoints.

## 🔧 How It Works

```
Query: "What are the pricing options?"
           │
           ▼
┌──────────────────────────┐
│   AdaptiveCrawler        │
│   (statistical/embedding)│
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Intelligent Link         │ → Follows top_k most relevant links
│ Selection                │
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Confidence Scoring       │ → Stops at confidence_threshold
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ get_relevant_content()   │ → Returns scored pages
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ DeepSeek Chat            │ → Generates answer
└──────────────────────────┘
```

## 📊 Strategies

### Statistical (Default)
- Fast, keyword-based matching
- Good for exact term searches
- Lower resource usage

### Embedding
- Semantic understanding
- Finds conceptually related content
- Better for natural language queries
- Requires more resources

## 🔗 n8n Integration

**HTTP Request Node Settings:**
- Method: `POST`
- URL: `https://your-app.zeabur.app/crawl`
- Body Content Type: `JSON`
- Timeout: `100000` (100 seconds)

**JSON Body:**
```json
{
  "start_url": "{{ $json.url }}",
  "query": "{{ $json.question }}",
  "max_pages": 15,
  "strategy": "statistical"
}
```

## 📝 Example Usage

### cURL
```bash
curl -X POST "https://crawl4ai-adaptive.zeabur.app/crawl" \
  -H "Content-Type: application/json" \
  -d '{
    "start_url": "https://docs.example.com",
    "query": "How do I authenticate?",
    "max_pages": 15
  }'
```

### Python
```python
import requests

response = requests.post(
    "https://crawl4ai-adaptive.zeabur.app/crawl",
    json={
        "start_url": "https://docs.example.com",
        "query": "How do I authenticate?",
        "max_pages": 15,
        "strategy": "statistical"
    },
    timeout=120
)

result = response.json()
print(result["answer"])
```

## ⚙️ Performance Notes

- **Typical response time**: 10-60 seconds depending on site complexity
- **Timeout recommendation**: 100+ seconds in n8n/clients
- **For faster results**: Use lower `max_pages` and `confidence_threshold`

## 📄 License

MIT
