---
title: "Tool Use and Function Calling: Giving Agents Real-World Capabilities"
date: 2026-06-17 00:00:00 +0300
categories: [AI Engineering, LLM]
tags: [agents, tool-use, function-calling, openai-tools, api-integration]
image:
  path: /assets/img/cover-agent-tool-calling.webp
  alt: Tool calling and function calling diagram showing agent-tool interaction
---

## Introduction

An agent without tools is just a fancy chatbot. **Tools are what connect LLMs to the real world** — databases, APIs, file systems, search engines, code executors. Without them, the model is limited to its training cutoff and parametric knowledge.

In this post, we'll move beyond the simple tool wrappers from Post 1 and build a production-grade tool system covering:

- **Tool schemas** compatible with OpenAI, Anthropic, and open models
- **Robust error handling** — retries, rate limiting, timeouts
- **Parallel tool execution** for performance
- **Tool composition** — chains and DAGs of tool calls
- **Security boundaries** — input validation and access control

## Anatomy of a Tool

Every tool needs three things:

1. **Name**: Unique identifier
2. **Description**: Clear explanation of what it does (critical for the LLM to choose correctly)
3. **Parameter Schema**: JSON Schema defining inputs

### OpenAI Function Calling Format

{% raw %}
```python
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Nairobi, Kenya'"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Query the product database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum rows to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```
{% endraw %}

### Anthropic Tool Format

Anthropic uses a slightly different schema — notice the lack of `"type": "function"` wrapper:

{% raw %}
```python
anthropic_tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
]
```
{% endraw %}

### Universal Tool Class

Let's build a unified tool definition that works across providers:

{% raw %}
```python
from typing import Any, Callable, Optional
import json
import time
import inspect


class Tool:
    """Unified tool definition compatible with OpenAI and Anthropic formats."""
    
    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: dict = None,
        max_retries: int = 2,
        timeout: float = 30.0,
        rate_limit: Optional[float] = None,  # Min seconds between calls
    ):
        self.name = name
        self.description = description
        self.handler = handler
        self.parameters = parameters or self._infer_parameters(handler)
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit = rate_limit
        self._last_call_time = 0.0
    
    def _infer_parameters(self, func: Callable) -> dict:
        """Auto-generate JSON Schema from function signature."""
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            # Map Python types to JSON Schema types
            type_map = {
                str: {"type": "string"},
                int: {"type": "integer"},
                float: {"type": "number"},
                bool: {"type": "boolean"},
                list: {"type": "array"},
                dict: {"type": "object"},
            }
            
            annotation = param.annotation
            schema_type = type_map.get(annotation, {"type": "string"})
            properties[param_name] = {
                **schema_type,
                "description": f"Parameter {param_name}"
            }
            
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def to_anthropic_format(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }
    
    def execute(self, **kwargs) -> dict:
        """Execute the tool with retry logic and rate limiting."""
        # Rate limiting
        if self.rate_limit:
            elapsed = time.time() - self._last_call_time
            if elapsed < self.rate_limit:
                wait = self.rate_limit - elapsed
                time.sleep(wait)
        
        # Retry loop
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                result = self.handler(**kwargs)
                duration = time.time() - start
                self._last_call_time = time.time()
                
                return {
                    "success": True,
                    "result": result,
                    "duration_ms": round(duration * 1000, 2),
                    "attempts": attempt + 1
                }
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                continue
        
        return {
            "success": False,
            "error": last_error,
            "duration_ms": 0,
            "attempts": self.max_retries + 1
        }
```
{% endraw %}

## Building Production Tools

Let's build three real-world tools with proper error handling.

### 1. Web Search Tool

{% raw %}
```python
import requests
from typing import Optional


class WebSearchTool(Tool):
    """Search the web using a search API."""
    
    def __init__(self, api_key: str, search_engine_id: str):
        super().__init__(
            name="web_search",
            description="Search the web for current information. Use this for real-time data, news, and facts.",
            handler=self._search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results (1-10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            rate_limit=1.0  # Max 1 call per second
        )
        self.api_key = api_key
        self.search_engine_id = search_engine_id
    
    def _search(self, query: str, num_results: int = 5) -> str:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(num_results, 10)
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "items" not in data:
            return f"No results found for: {query}"
        
        results = []
        for item in data["items"][:num_results]:
            results.append(f"- [{item['title']}]({item['link']}): {item.get('snippet', '')}")
        
        return f"Search results for '{query}':\n" + "\n".join(results)
```
{% endraw %}

### 2. Calculator with Expression Safety

{% raw %}
```python
import math
import ast
import operator


class SafeCalculatorTool(Tool):
    """Evaluate mathematical expressions with security constraints."""
    
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate a mathematical expression. Supports +, -, *, /, **, %, and parentheses.",
            handler=self._calculate,
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        )
    
    def _calculate(self, expression: str) -> str:
        # Parse the expression into an AST
        try:
            tree = ast.parse(expression.strip(), mode='eval')
        except SyntaxError:
            return "Error: Invalid syntax"
        
        # Evaluate safely using only allowed operators
        def _eval(node):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Unsupported constant: {type(node.value)}")
            
            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type not in self.ALLOWED_OPERATORS:
                    raise ValueError(f"Operator not allowed: {op_type.__name__}")
                left = _eval(node.left)
                right = _eval(node.right)
                return self.ALLOWED_OPERATORS[op_type](left, right)
            
            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type not in self.ALLOWED_OPERATORS:
                    raise ValueError(f"Operator not allowed: {op_type.__name__}")
                operand = _eval(node.operand)
                return self.ALLOWED_OPERATORS[op_type](operand)
            
            elif isinstance(node, ast.Call):
                # Allow math functions via math.<func>()
                if isinstance(node.func, ast.Attribute) and \
                   isinstance(node.func.value, ast.Name) and \
                   node.func.value.id == "math":
                    func_name = node.func.attr
                    if hasattr(math, func_name) and callable(getattr(math, func_name)):
                        args = [_eval(arg) for arg in node.args]
                        return getattr(math, func_name)(*args)
                raise ValueError(f"Function calls not allowed: {ast.dump(node)}")
            
            else:
                raise ValueError(f"Expression type not allowed: {type(node).__name__}")
        
        try:
            result = _eval(tree)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
```
{% endraw %}

### 3. API Integration Tool

{% raw %}
```python
class APITool(Tool):
    """Generic HTTP API integration tool."""
    
    def __init__(self, name: str, description: str,
                 base_url: str, endpoint: str,
                 method: str = "GET", headers: dict = None,
                 params_schema: dict = None):
        
        self.base_url = base_url
        self.endpoint = endpoint
        self.method = method.upper()
        self.headers = headers or {}
        
        super().__init__(
            name=name,
            description=description,
            handler=self._call_api,
            parameters=params_schema or {
                "type": "object",
                "properties": {},
                "required": []
            },
            timeout=15.0
        )
    
    def _call_api(self, **kwargs) -> str:
        url = f"{self.base_url}{self.endpoint}"
        
        if self.method == "GET":
            response = requests.get(
                url, params=kwargs, headers=self.headers, timeout=self.timeout
            )
        elif self.method == "POST":
            response = requests.post(
                url, json=kwargs, headers=self.headers, timeout=self.timeout
            )
        else:
            return f"Error: Unsupported method {self.method}"
        
        response.raise_for_status()
        
        # Truncate long responses
        text = response.text
        if len(text) > 5000:
            text = text[:5000] + "... [truncated]"
        
        return text
```
{% endraw %}

## Parallel Tool Execution

When an agent needs multiple independent tools called, sequential execution wastes time. **Parallel execution** dramatically reduces latency:

{% raw %}
```python
from concurrent.futures import ThreadPoolExecutor, as_completed


class ToolExecutor:
    """Manages tool execution with parallel support."""
    
    def __init__(self, tools: list[Tool], max_parallel: int = 5):
        self.tools = {t.name: t for t in tools}
        self.max_parallel = max_parallel
    
    def execute_all(self, tool_calls: list[dict]) -> dict:
        """Execute multiple tools in parallel when possible.
        
        tool_calls: [{"name": "web_search", "args": {"query": "..."}}, ...]
        """
        results = {}
        independent_calls = []
        dependent_calls = []
        
        # For simplicity, treat all as independent here
        # In production, build a dependency DAG
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {}
            for call in tool_calls:
                if call["name"] in self.tools:
                    future = executor.submit(
                        self.tools[call["name"]].execute,
                        **call["args"]
                    )
                    futures[future] = call["name"]
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = {
                        "success": False,
                        "error": str(e),
                        "duration_ms": 0,
                        "attempts": 1
                    }
        
        return results
```
{% endraw %}

## Error Handling Patterns

### Exponential Backoff

{% raw %}
```python
import random
import time


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0):
    """Calculate wait time with jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter


class ResilientTool(Tool):
    """Tool wrapper with automatic retry and backoff."""
    
    def execute_with_backoff(self, **kwargs) -> dict:
        for attempt in range(self.max_retries + 1):
            result = super().execute(**kwargs)
            if result["success"]:
                return result
            
            if attempt < self.max_retries:
                wait = exponential_backoff(attempt)
                time.sleep(wait)
        
        return result
```
{% endraw %}

### Graceful Degradation

When a tool fails, the agent should adapt, not crash:

{% raw %}
```python
def execute_with_fallback(primary_tool: Tool, fallback_tool: Tool, **kwargs) -> dict:
    """Try primary tool, fall back to alternative on failure."""
    result = primary_tool.execute(**kwargs)
    if result["success"]:
        return result
    
    # Log the failure
    print(f"Primary tool '{primary_tool.name}' failed: {result.get('error')}")
    print(f"Falling back to '{fallback_tool.name}'")
    
    return fallback_tool.execute(**kwargs)
```
{% endraw %}

## Tool Composition Patterns

### Chain (Sequential)

Output of one tool becomes input to the next:

{% raw %}
```python
def tool_chain(tools: list[tuple[Tool, dict]], 
               context_passthrough: bool = True) -> list[dict]:
    """Execute tools sequentially, passing results through context."""
    context = {}
    results = []
    
    for tool, params_template in tools:
        # Resolve parameters from context if needed
        params = {}
        for key, value in params_template.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to previous result: $tool_name.field
                ref = value[1:].split(".")
                source = results if context_passthrough else {}
                # Look through results for the reference
                resolved = resolve_reference(ref, context, results)
                params[key] = resolved
            else:
                params[key] = value
        
        result = tool.execute(**params)
        results.append(result)
        context[tool.name] = result.get("result", "")
    
    return results
```
{% endraw %}

### Router (Conditional)

The LLM decides which tool to call based on the input:

{% raw %}
```python
def route_to_tool(user_input: str, tools: dict[str, Tool], 
                  classifier_llm) -> Tool:
    """Use an LLM to classify the input and route to the right tool."""
    prompt = f"""Given the user request, select the most appropriate tool.
Available tools: {', '.join(tools.keys())}
Tool descriptions: {', '.join(t.description for t in tools.values())}

User: {user_input}
Selected tool:"""
    
    response = classifier_llm.complete(prompt, max_tokens=20)
    tool_name = response.text.strip()
    return tools.get(tool_name, tools["default"])
```
{% endraw %}

## Production Checklist

| Concern | Implementation |
|---------|---------------|
| **Timeouts** | Every tool call needs a hard timeout |
| **Retries** | 2-3 retries with exponential backoff |
| **Rate limiting** | API-level and tool-level rate limits |
| **Input validation** | Schema validation before execution |
| **Output truncation** | Cap tool output to prevent context overflow |
| **Idempotency** | Tool calls should be safe to retry |
| **Cost tracking** | Log token usage per tool call |
| **Audit trail** | Every tool invocation logged |

## Conclusion

Tools are the bridge between LLM reasoning and real-world action. A well-designed tool system — with proper schemas, error handling, rate limiting, and composition — is the difference between a demo agent and a production system.

Key takeaways:

- **Descriptions matter**: The LLM chooses tools based on your descriptions. Be clear and specific.
- **Fail gracefully**: Tools fail. Plan for it with retries, backoffs, and fallbacks.
- **Parallelize wisely**: Independent tools should run in parallel; dependent ones in sequence.
- **Validate everything**: Never trust LLM-generated tool arguments. Validate against schemas before execution.

**In the next post**, we'll scale from single agents to multi-agent systems, orchestrating teams of specialized agents with LangGraph and CrewAI.

## Further Reading

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Building AI Agents: ReAct Pattern](/posts/agent-fundamentals/)
- [Memory Systems for AI Agents](/posts/agent-memory-systems/)
