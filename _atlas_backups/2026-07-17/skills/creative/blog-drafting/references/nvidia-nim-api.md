# NVIDIA NIM API — Free Tier Reference

## Sign Up
- Go to https://build.nvidia.com/settings/api-keys
- Sign up for free (no credit card required)
- Generate an API key

## Rate Limits
- **40 requests per minute (RPM)** for free tier
- No per-token billing — the free tier is rate-limit gated, not token-gated
- Can request increase to 200 RPM via NVIDIA Developer Forums

## Available Free Models (as of 2026)
Notable models among ~46 available:
- Llama 4 Maverick (Meta) — most popular
- Mistral Large 3 675B
- Qwen3 Coder 480B — agentic coding
- MiniMax M2.7 230B — competes with Claude on coding
- DeepSeek models
- Google Gemma
- NVIDIA Nemotron series

## Python Example
```python
import requests

API_KEY = "nvapi-..."
url = "https://build.nvidia.com/mistralai/mistral-large-3/v1/chat/completions"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

payload = {
    "model": "mistralai/mistral-large-3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 500
}

response = requests.post(url, json=payload, headers=headers)
print(response.json()["choices"][0]["message"]["content"])
```

## Usage in Blog Content
Mention NVIDIA NIM in posts about:
- Free alternatives to paid LLM APIs
- Prototyping RAG pipelines at low cost
- Running open-source models without GPU infrastructure
- Graph RAG tutorials (use NIM as the LLM backend instead of OpenAI)
