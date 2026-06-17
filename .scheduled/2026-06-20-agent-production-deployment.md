---
title: "Deploying AI Agents to Production: Scaling, Error Handling, and State Management"
date: 2026-06-20 00:00:00 +0300
categories: [AI Engineering, LLM]
tags: [agents, production, deployment, scaling, state-management]
image:
  path: /assets/img/cover-agent-production-deployment.webp
  alt: Agent production deployment architecture showing queues, workers, and storage
---

## Introduction

Building an agent in a Jupyter notebook is easy. Running it reliably in production — handling hundreds of concurrent users, surviving API outages, persisting state across restarts, and scaling cost-effectively — is an entirely different challenge.

**The gap between a working agent and a production agent is infrastructure.**

In this post, we'll transform our ReAct agent into a production-ready service covering:

- **Queue-based architecture** — decoupling request receipt from agent execution
- **Error handling** — graceful degradation when LLMs or tools fail
- **State persistence** — surviving process restarts without losing context
- **Rate limiting** — protecting downstream APIs and managing costs
- **Scaling** — horizontal scaling with worker pools
- **Caching** — avoiding redundant LLM calls

## Architecture Overview

A production agent system separates concerns into layers:

```
[User] → [API Gateway] → [Task Queue (Redis)] → [Worker Pool] → [LLM APIs]
                                    ↓                              ↓
                              [State Store (PostgreSQL)]    [Tool APIs]
```

**Why a queue?** LLM calls can take 2-30 seconds. If you handle requests synchronously, one slow agent blocks all others. A queue lets you:

1. Accept requests immediately
2. Process them asynchronously
3. Poll or webhook results back to users

## Queue-Based Architecture with Redis + Celery

[Celery](https://docs.celeryq.dev/) is a distributed task queue that works beautifully with Redis as a broker.

### Setup

```bash
pip install celery redis
```

{% raw %}
```python
# celery_app.py
from celery import Celery

celery_app = Celery(
    "agent_worker",
    broker="redis://localhost:6379/0",  # Task queue
    backend="redis://localhost:6379/1"  # Result store
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Africa/Nairobi",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # Re-queue on worker crash
    worker_prefetch_multiplier=1,  # One task at a time per worker
)
```
{% endraw %}

### Defining Agent Tasks

{% raw %}
```python
# tasks.py
from celery import current_task
from .celery_app import celery_app
from .agent import ReActAgent
from .state import SessionState
import time


@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def run_agent(self, session_id: str, user_message: str, 
              model: str = "gpt-4o"):
    """
    Celery task to run an agent.
    Tracks progress and handles retries.
    """
    try:
        # Update task state so clients can poll progress
        self.update_state(
            state="PROGRESS",
            meta={"step": "initializing", "session_id": session_id}
        )
        
        # Load session state
        state_manager = SessionState()
        session = state_manager.load(session_id)
        
        # Run the agent
        agent = ReActAgent(model=model)
        
        self.update_state(
            state="PROGRESS",
            meta={"step": "processing", "session_id": session_id}
        )
        
        start = time.time()
        result = agent.run(user_message, session)
        duration = time.time() - start
        
        # Save updated state
        state_manager.save(session)
        
        # Log the execution
        state_manager.log_execution(
            session_id=session_id,
            user_message=user_message,
            agent_result=result,
            duration_ms=round(duration * 1000, 2),
            model=model,
            total_tokens=agent.last_token_usage
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "result": result,
            "duration_ms": round(duration * 1000, 2)
        }
        
    except Exception as exc:
        # Log the failure
        state_manager = SessionState()
        state_manager.log_error(session_id, str(exc))
        
        # Retry with backoff
        raise self.retry(exc=exc)
```
{% endraw %}

### The API Layer

{% raw %}
```python
# api.py
from flask import Flask, request, jsonify
from celery.result import AsyncResult
from .celery_app import celery_app
from .tasks import run_agent
import uuid

app = Flask(__name__)


@app.route("/agent/run", methods=["POST"])
def start_agent():
    """Start an agent run, return immediately with a task ID."""
    data = request.json
    session_id = data.get("session_id", str(uuid.uuid4()))
    message = data.get("message", "")
    model = data.get("model", "gpt-4o")
    
    # Queue the task
    task = run_agent.delay(session_id, message, model)
    
    return jsonify({
        "task_id": task.id,
        "session_id": session_id,
        "status": "queued"
    }), 202


@app.route("/agent/status/<task_id>", methods=["GET"])
def get_status(task_id):
    """Poll for agent task status."""
    task = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task.state,
    }
    
    if task.state == "PENDING":
        response["info"] = "Task is queued"
    elif task.state == "PROGRESS":
        response["info"] = task.info
    elif task.state == "SUCCESS":
        response["result"] = task.result
    elif task.state == "FAILURE":
        response["error"] = str(task.info)
    
    return jsonify(response)


if __name__ == "__main__":
    app.run(port=8000)
```
{% endraw %}

### Starting Workers

```bash
# Terminal 1: Start Redis (if not running)
redis-server

# Terminal 2: Start Celery worker
celery -A tasks worker --loglevel=info --concurrency=4

# Terminal 3: Start API
python api.py
```

## State Persistence Across Restarts

Agents must survive worker restarts. Store the full conversation state in PostgreSQL:

{% raw %}
```python
import psycopg2
import json
from datetime import datetime
import pickle


class SessionState:
    """Persist agent state to PostgreSQL."""
    
    def __init__(self, connection_string: str = None):
        self.conn = psycopg2.connect(
            connection_string or 
            "postgresql://user:pass@localhost:5432/agent_state"
        )
        self._create_tables()
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                state_data BYTEA,  -- Pickled agent state
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_logs (
                id SERIAL PRIMARY KEY,
                session_id TEXT REFERENCES sessions(session_id),
                user_message TEXT,
                agent_result TEXT,
                duration_ms FLOAT,
                model TEXT,
                total_tokens INT,
                error TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        self.conn.commit()
    
    def save(self, session: dict):
        """Save agent session state."""
        cursor = self.conn.cursor()
        pickled = pickle.dumps(session)
        
        cursor.execute("""
            INSERT INTO sessions (session_id, user_id, state_data, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (session_id) 
            DO UPDATE SET state_data = EXCLUDED.state_data,
                          updated_at = NOW()
        """, (session["session_id"], session.get("user_id"), pickled))
        self.conn.commit()
    
    def load(self, session_id: str) -> dict:
        """Load agent session state."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT state_data FROM sessions WHERE session_id = %s",
            (session_id,)
        )
        row = cursor.fetchone()
        if row:
            return pickle.loads(row[0])
        
        # Return default state for new sessions
        return {
            "session_id": session_id,
            "messages": [],
            "context": {},
            "created_at": datetime.now().isoformat()
        }
    
    def log_execution(self, **kwargs):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO execution_logs 
            (session_id, user_message, agent_result, duration_ms, 
             model, total_tokens)
            VALUES (%(session_id)s, %(user_message)s, %(agent_result)s,
                    %(duration_ms)s, %(model)s, %(total_tokens)s)
        """, kwargs)
        self.conn.commit()
```
{% endraw %}

## Error Handling and Graceful Degradation

### Handling LLM API Outages

{% raw %}
```python
import time
import random
from typing import Optional


class ResilientLLM:
    """LLM client with circuit breaker and fallback models."""
    
    def __init__(self):
        self.models = [
            {"name": "gpt-4o", "provider": "openai"},
            {"name": "gpt-4o-mini", "provider": "openai"},
            {"name": "claude-3-5-sonnet", "provider": "anthropic"},
        ]
        self.model_status = {m["name"]: {"healthy": True, "cooldown_until": 0} 
                            for m in self.models}
    
    def complete(self, messages: list, preferred_model: str = "gpt-4o") -> str:
        """Try models in order of preference, falling back on failure."""
        
        # Sort by preference
        ordered_models = sorted(
            self.models,
            key=lambda m: 0 if m["name"] == preferred_model else 1
        )
        
        for model in ordered_models:
            name = model["name"]
            status = self.model_status[name]
            
            # Skip models in cooldown
            if not status["healthy"] and time.time() < status["cooldown_until"]:
                continue
            
            try:
                result = self._call_model(name, messages)
                # Reset health on success
                status["healthy"] = True
                return result
            except Exception as e:
                # Mark as unhealthy with exponential backoff
                cooldown = min(300, (2 ** random.randint(0, 3)) * 10)
                status["healthy"] = False
                status["cooldown_until"] = time.time() + cooldown
                
                print(f"Model {name} failed: {e}. Cooling down for {cooldown}s")
                continue
        
        raise Exception("All models unavailable")
    
    def _call_model(self, model_name: str, messages: list) -> str:
        """Actual API call - implement per-provider."""
        if "gpt" in model_name:
            return self._call_openai(model_name, messages)
        else:
            return self._call_anthropic(model_name, messages)
```
{% endraw %}

### Idempotency for Tool Calls

Tool calls must be safe to retry. Implement idempotency keys:

{% raw %}
```python
import hashlib


class IdempotentToolExecutor:
    """Prevents duplicate execution of the same tool call."""
    
    def __init__(self, cache_ttl: int = 3600):
        self.cache = {}  # key -> result
        self.cache_ttl = cache_ttl
    
    def _make_key(self, tool_name: str, args: dict) -> str:
        """Generate a deterministic key from tool call."""
        serialized = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def execute(self, tool, **kwargs) -> dict:
        key = self._make_key(tool.name, kwargs)
        
        # Return cached result if available
        if key in self.cache:
            cached = self.cache[key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["result"]
        
        # Execute and cache
        result = tool.execute(**kwargs)
        self.cache[key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        return result
```
{% endraw %}

## Rate Limiting

Protect downstream APIs and control costs:

{% raw %}
```python
import time
from collections import defaultdict
import asyncio


class RateLimiter:
    """Token bucket rate limiter per API."""
    
    def __init__(self, rpm: int = 60):  # Requests per minute
        self.rpm = rpm
        self.tokens = rpm
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        refill = elapsed * (self.rpm / 60.0)
        self.tokens = min(self.rpm, self.tokens + refill)
        self.last_refill = now
    
    async def acquire(self) -> float:
        """Wait for a token and return the wait time."""
        async with self.lock:
            await self._refill()
            if self.tokens < 1:
                wait = (1 - self.tokens) / (self.rpm / 60.0)
                await asyncio.sleep(wait)
                await self._refill()
            self.tokens -= 1
            return 0.0


class MultiAPI RateLimiter:
    """Rate limit per API provider."""
    
    def __init__(self):
        self.limiters = {
            "openai": RateLimiter(rpm=500),     # 500 RPM for OpenAI
            "anthropic": RateLimiter(rpm=100),   # 100 RPM for Anthropic
            "google": RateLimiter(rpm=60),       # 60 RPM for Google
            "serpapi": RateLimiter(rpm=30),      # 30 RPM for SerpAPI
        }
    
    async def wait_for(self, provider: str):
        if provider in self.limiters:
            await self.limiters[provider].acquire()
```
{% endraw %}

## Cost Management

{% raw %}
```python
class CostTracker:
    """Track and cap agent costs."""
    
    def __init__(self, daily_budget: float = 50.0, 
                 session_budget: float = 2.0):
        self.daily_budget = daily_budget
        self.session_budget = session_budget
        self.daily_spend = 0.0
        self.session_spend = 0.0
        self.reset_time = time.time() + 86400
    
    def check_budget(self, session_id: str) -> bool:
        """Return True if within budget, False if exceeded."""
        now = time.time()
        if now > self.reset_time:
            self.daily_spend = 0.0
            self.reset_time = now + 86400
        
        if self.daily_spend >= self.daily_budget:
            return False
        if self.session_spend >= self.session_budget:
            return False
        return True
    
    def record_llm_call(self, model: str, input_tokens: int, 
                        output_tokens: int, session_id: str):
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        self.daily_spend += cost
        self.session_spend += cost
    
    def end_session(self):
        self.session_spend = 0.0
```
{% endraw %}

## Caching LLM Responses

Many agent requests are repetitive. Cache deterministic LLM calls:

{% raw %}
```python
import hashlib
import redis


class LLMCache:
    """Cache LLM responses for identical inputs."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/2",
                 ttl: int = 3600):
        self.client = redis.from_url(redis_url)
        self.ttl = ttl
    
    def _make_key(self, model: str, messages: list, temperature: float) -> str:
        """Generate cache key from the full request."""
        content = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature
        }, sort_keys=True)
        return f"llm_cache:{hashlib.sha256(content.encode()).hexdigest()}"
    
    def get(self, model: str, messages: list, 
            temperature: float = 0) -> Optional[str]:
        if temperature > 0:
            return None  # Don't cache non-deterministic calls
        
        key = self._make_key(model, messages, temperature)
        cached = self.client.get(key)
        return cached.decode() if cached else None
    
    def set(self, model: str, messages: list, response: str,
            temperature: float = 0):
        if temperature > 0:
            return
        
        key = self._make_key(model, messages, temperature)
        self.client.setex(key, self.ttl, response)
```
{% endraw %}

## Production Checklist

| Concern | Solution |
|---------|----------|
| **Concurrent users** | Queue-based architecture with Celery |
| **Worker crashes** | `task_acks_late=True` re-queues tasks |
| **LLM API down** | Circuit breaker + fallback models |
| **Rate limits** | Token bucket rate limiter per API |
| **Cost spikes** | Daily + session budget caps |
| **Duplicate calls** | Idempotency keys with TTL |
| **Slow start** | Model warm-up on worker boot |
| **Memory leaks** | Restart workers every N tasks |
| **State loss** | PostgreSQL persistence with pickle |
| **Latency** | LLM response caching (temp=0) |

## Conclusion

Deploying agents to production requires rethinking the architecture: queues for async processing, persistent state for reliability, circuit breakers for resilience, and budgets for cost control.

The key principle: **design for failure.** LLM APIs go down. Workers crash. Tools time out. A production agent system absorbs these failures gracefully, retries intelligently, and never loses user state.

**In the final post of this series**, we'll cover the most critical topic for production agents: security — guardrails, input validation, sandboxing, and prompt injection defense.

## Further Reading

- [Celery Documentation](https://docs.celeryq.dev/)
- [Redis as a Task Queue](https://redis.io/glossary/redis-queue/)
- [Agent Observability](/posts/agent-observability/)
- [Building Secure AI Agents](/posts/agent-security/)
