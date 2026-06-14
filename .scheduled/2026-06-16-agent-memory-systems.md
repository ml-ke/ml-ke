---
title: "Memory Systems for AI Agents: Short-Term, Long-Term, and Persistent Storage"
date: 2026-06-16 00:00:00 +0300
categories: [AI Engineering, LLM]
tags: [agents, memory-systems, vector-databases, chromadb, persistent-memory]
image:
  path: /assets/img/cover-agent-memory-systems.webp
  alt: AI Agent memory systems architecture diagram showing short-term and long-term storage
---

## Introduction

The agent we built in the [previous post]({% post_url 2026-06-15-agent-fundamentals %}) is stateless. Each conversation starts fresh — no memory of past interactions, no learning from previous runs, no persistent knowledge. In production, that's unacceptable.

**Memory is what transforms a one-shot LLM call into a learning system.** It enables agents to:

- Remember context across a conversation (short-term memory)
- Recall facts and patterns from past sessions (long-term memory)
- Store user preferences and learned behaviors (persistent memory)
- Retrieve relevant information without re-processing everything (retrieval-augmented generation)

In this post, we'll build three memory systems from scratch — conversation buffers, vector-based long-term memory, and SQLite-backed persistent storage — and learn when to use each.

## Short-Term Memory: Conversation Buffers

The simplest form of agent memory is the **conversation buffer** — a sliding window of recent messages. The LLM sees the last N turns and uses them as context.

### Sliding Window Buffer

{% raw %}
```python
from collections import deque
from typing import List, Dict


class SlidingWindowBuffer:
    """Keeps the last N messages in conversation context."""
    
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages: deque[Dict] = deque(maxlen=max_messages)
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def add_tool_result(self, tool_name: str, result: str):
        self.messages.append({
            "role": "tool",
            "content": str(result),
            "tool_call_id": tool_name
        })
    
    def get_context(self) -> List[Dict]:
        return list(self.messages)
    
    def clear(self):
        self.messages.clear()
```
{% endraw %}

**Pros**: Simple, low latency, predictable token usage.  
**Cons**: Drops older context completely — the agent "forgets" anything beyond the window.

### Token-Limited Buffer

A smarter approach: trim by token count instead of message count, ensuring the full context fits within the model's limit:

{% raw %}
```python
import tiktoken


class TokenLimitedBuffer:
    """Keeps as many recent messages as fit within a token budget."""
    
    def __init__(self, max_tokens: int = 4000, model: str = "gpt-4o"):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)
        self.messages: List[Dict] = []
    
    def _count_tokens(self, messages: List[Dict]) -> int:
        return len(self.encoding.encode(str(messages)))
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
        # Trim from the front until within budget
        while self._count_tokens(self.messages) > self.max_tokens:
            if len(self.messages) <= 1:
                break
            self.messages.pop(0)
    
    def get_context(self) -> List[Dict]:
        return self.messages
```
{% endraw %}

### Summarization Memory

For long conversations, even a token-limited buffer eventually drops important early context. **Summarization memory** periodically compresses older messages into a summary:

{% raw %}
```python
class SummarizationMemory:
    """Compresses old messages into summaries to preserve context."""
    
    def __init__(self, llm_client, max_tokens: int = 3000):
        self.llm_client = llm_client
        self.max_tokens = max_tokens
        self.summary = ""
        self.recent_messages: List[Dict] = []
    
    def add_message(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})
        self._maybe_summarize()
    
    def _maybe_summarize(self):
        """If recent messages are too long, summarize old ones."""
        total = len(str(self.recent_messages))
        if total > self.max_tokens * 4:  # Arbitrary threshold
            self._compress()
    
    def _compress(self):
        # Take oldest messages and summarize them
        old_messages = self.recent_messages[:len(self.recent_messages) // 2]
        new_messages = self.recent_messages[len(self.recent_messages) // 2:]
        
        prompt = f"""Summarize the following conversation so far, preserving
key facts, user preferences, decisions made, and any important context.
Current summary (if any): {self.summary}

Messages to compress:
{old_messages}

New summary:"""
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        self.summary = response.choices[0].message.content
        self.recent_messages = new_messages
    
    def get_context(self) -> List[Dict]:
        context = []
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Conversation summary so far: {self.summary}"
            })
        context.extend(self.recent_messages)
        return context
```
{% endraw %}

## Long-Term Memory: Vector Databases

Short-term memory is lost when the session ends. **Long-term memory** persists useful information — facts, insights, user preferences — across sessions. Vector databases are the standard approach: embed information into vectors and retrieve relevant chunks via semantic similarity.

### Using ChromaDB for Long-Term Memory

Let's build a memory layer using ChromaDB:

{% raw %}
```python
import chromadb
from chromadb.utils import embedding_functions


class VectorMemory:
    """Long-term memory using vector embeddings and ChromaDB."""
    
    def __init__(self, collection_name: str = "agent_memory",
                 persist_directory: str = "./agent_memory_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
        except ValueError:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
    
    def store(self, content: str, metadata: dict = None, id: str = None):
        """Store a memory with optional metadata."""
        if id is None:
            import uuid
            id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        
        # Add timestamp automatically
        from datetime import datetime
        metadata["timestamp"] = datetime.now().isoformat()
        
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[id]
        )
    
    def retrieve(self, query: str, n_results: int = 5, 
                  filter_metadata: dict = None) -> List[dict]:
        """Retrieve relevant memories by semantic similarity."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata  # Optional metadata filter
        )
        
        memories = []
        for i in range(len(results['ids'][0])):
            memories.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return memories
    
    def update(self, id: str, content: str, metadata: dict = None):
        """Update an existing memory."""
        self.collection.update(
            documents=[content],
            metadatas=[metadata] if metadata else None,
            ids=[id]
        )
    
    def delete_old(self, max_age_days: int = 30):
        """Clean up memories older than max_age_days."""
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        self.collection.delete(
            where={"timestamp": {"$lt": cutoff}}
        )
```
{% endraw %}

### Integrating Memory into an Agent

Here's how the memory system plugs into our ReAct agent from the previous post:

{% raw %}
```python
class AgentWithMemory:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.short_term = SummarizationMemory(llm_client)
        self.long_term = VectorMemory()
        self.session_id = str(uuid.uuid4())
    
    def process_message(self, user_message: str) -> str:
        # 1. Retrieve relevant long-term memories
        memories = self.long_term.retrieve(user_message, n_results=3)
        memory_context = ""
        if memories:
            memory_context = "Relevant past memories:\n" + "\n".join(
                f"- {m['content']}" for m in memories
            )
        
        # 2. Build context with short-term + long-term memory
        context = self.short_term.get_context()
        if memory_context:
            context.insert(0, {
                "role": "system",
                "content": memory_context
            })
        
        # 3. Process with the agent loop
        # (Run the ReAct loop from Post 1 here...)
        
        # 4. Store important information in long-term memory
        self._extract_and_store(user_message, response)
        
        return response
    
    def _extract_and_store(self, user_msg: str, response: str):
        """Decide what to remember for the long term."""
        # Simple heuristic: store facts that mention user preferences
        preference_keywords = ["prefer", "like", "want", "don't", "my",
                               "favorite", "always", "never"]
        for keyword in preference_keywords:
            if keyword in user_msg.lower():
                self.long_term.store(
                    content=f"User said: {user_msg} | Agent responded: {response}",
                    metadata={
                        "session": self.session_id,
                        "type": "preference"
                    }
                )
                break
```
{% endraw %}

## Persistent Storage: SQLite Backend

Vector databases are great for semantic retrieval, but sometimes you need **structured, queryable persistent storage** — user accounts, session logs, tool execution history, cost tracking.

{% raw %}
```python
import sqlite3
import json
from datetime import datetime


class PersistentStore:
    """SQLite-backed persistent storage for agent state."""
    
    def __init__(self, db_path: str = "./agent_store.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT
            )
        """)
        
        # Session state (key-value store per session)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT,
                key TEXT,
                value TEXT,
                updated_at TEXT,
                PRIMARY KEY (session_id, key),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # Tool execution log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                tool_name TEXT,
                input_params TEXT,
                output TEXT,
                success BOOLEAN,
                duration_ms REAL,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        self.conn.commit()
    
    def create_session(self, session_id: str, user_id: str = "anonymous",
                       metadata: dict = None):
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, now, now, json.dumps(metadata or {}))
        )
        self.conn.commit()
    
    def set_state(self, session_id: str, key: str, value: str):
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO session_state VALUES (?, ?, ?, ?)",
            (session_id, key, value, now)
        )
        self.conn.commit()
    
    def get_state(self, session_id: str, key: str) -> str:
        cursor = self.conn.execute(
            "SELECT value FROM session_state WHERE session_id = ? AND key = ?",
            (session_id, key)
        )
        row = cursor.fetchone()
        return row[0] if row else None
    
    def log_tool_call(self, session_id: str, tool_name: str,
                      input_params: dict, output: str,
                      success: bool, duration_ms: float):
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO tool_logs (session_id, tool_name, input_params, "
            "output, success, duration_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, tool_name, json.dumps(input_params),
             str(output), success, duration_ms, now)
        )
        self.conn.commit()
    
    def get_session_history(self, session_id: str) -> List[dict]:
        cursor = self.conn.execute(
            "SELECT * FROM tool_logs WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
```
{% endraw %}

## Comparing Memory Approaches

| Type | Persistence | Retrieval | Use Case | Example |
|------|-------------|-----------|----------|---------|
| **Sliding Window** | Ephemeral | Sequential | Current conversation context | Last 10 messages |
| **Summarization** | Ephemeral | Sequential | Long conversations | Chat session >1hr |
| **Vector Memory** | Persistent | Semantic | Cross-session recall | "Remember I prefer Python" |
| **SQLite Store** | Persistent | Structured | Logs, config, state | "Last week's tool usage" |
| **Hybrid** | Both | Both | Production agents | All of the above |

## When to Use Each

- **Short-term buffer only**: Simple question-answering, stateless chatbots
- **Short-term + summarization**: Customer support conversations, tutoring agents
- **Vector memory**: Research agents that need to accumulate knowledge over time
- **Full stack (all three)**: Production agents that learn user preferences, maintain sessions, and need debuggable history

## Practical Tips

1. **Embedding choice matters**: OpenAI's `text-embedding-3-small` is cost-effective for most agent memory. For domain-specific data, consider fine-tuned embeddings.
2. **Memory pruning**: Vector databases accumulate noise. Schedule regular cleanup of low-relevance or old memories.
3. **Metadata filtering**: Use metadata (session ID, timestamp, type) to scope retrievals — don't search the entire database for every query.
4. **Context window management**: Budget tokens carefully. A common split is 20% summary, 30% retrieved memories, 50% recent conversation.

## Conclusion

Memory is what separates a toy agent from a production system. By layering short-term buffers, vector-based long-term memory, and SQLite persistent storage, you create an agent that learns over time, maintains context, and can be debugged and monitored.

**In the next post**, we'll dive deep into tool use and function calling — how to define robust tool schemas, handle errors gracefully, and compose tools into powerful workflows.

## Further Reading

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAG vs. Fine-Tuning for LLMs](https://ml.co.ke/posts/rag-vs-fine-tuning)
- [Building AI Agents: ReAct Pattern]({% post_url 2026-06-15-agent-fundamentals %})
- [Tool Use and Function Calling]({% post_url 2026-06-17-agent-tool-calling %})
