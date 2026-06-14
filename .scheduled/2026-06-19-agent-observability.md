---
title: "Agent Observability and Debugging: Tracing, Logging, and Monitoring"
date: 2026-06-19 00:00:00 +0300
categories: [AI Engineering, LLM]
tags: [agents, observability, langfuse, tracing, debugging]
image:
  path: /assets/img/cover-agent-observability.webp
  alt: Agent observability dashboard showing traces, logs, and cost monitoring
---

## Introduction

Debugging an AI agent is fundamentally harder than debugging traditional software. Consider:

- **Non-deterministic**: The same input can produce different agent trajectories
- **Multi-step**: A single user request may trigger 5-15 LLM calls, tool executions, and reasoning loops
- **Cost opaque**: Which step cost the most tokens? Where did the agent get stuck?
- **Failure cascade**: An early bad decision poisons every subsequent step

Without observability, you're flying blind. When an agent produces a wrong answer, you have no way to tell if it was a bad tool call, a hallucination, a prompt issue, or a rate limit error.

In this post, we'll build an observability stack for AI agents covering:

- **Structured logging** — every agent decision recorded
- **Tracing** — end-to-end trace of a user request through the agent loop
- **Cost tracking** — per-step and per-session token usage
- **Replay** — re-running agent trajectories after fixes
- **Monitoring** — alerts for anomalous agent behavior

## Why Agent Debugging Sucks

Let's compare traditional debugging vs. agent debugging:

| Aspect | Traditional Code | AI Agent |
|--------|-----------------|----------|
| **Determinism** | Same input = same output | Same input = different output |
| **State** | Explicit variables + stack | Implicit in LLM context window |
| **Failures** | Crashes with stack trace | Silent wrong answers |
| **Reproduction** | Easy — known inputs | Hard — depends on LLM state |
| **Cost** | CPU/memory metrics | Token usage + API latency |

## Building an Agent Logger

Let's start with the foundation: capturing every step an agent takes.

{% raw %}
```python
import json
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class AgentStep:
    """A single step in an agent's execution."""
    step_number: int
    agent_name: str
    step_type: str  # 'thought', 'action', 'observation', 'tool_call', 'llm_call'
    input_data: Any
    output_data: Any
    token_usage: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentTrace:
    """Complete trace of an agent session."""
    trace_id: str
    session_id: str
    user_input: str
    final_output: str
    steps: list[AgentStep] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    success: bool = True


class AgentLogger:
    """Structured logger for agent execution."""
    
    # Approximate cost per 1K tokens
    MODEL_COSTS = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }
    
    def __init__(self, log_dir: str = "./agent_logs"):
        self.log_dir = log_dir
        import os
        os.makedirs(log_dir, exist_ok=True)
        self.current_trace: Optional[AgentTrace] = None
    
    def start_trace(self, session_id: str, user_input: str) -> str:
        trace_id = str(uuid.uuid4())[:8]
        self.current_trace = AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            user_input=user_input,
            start_time=time.time()
        )
        return trace_id
    
    def log_step(self, step: AgentStep):
        if self.current_trace:
            self.current_trace.steps.append(step)
            self.current_trace.total_tokens += sum(
                step.token_usage.get(k, 0) for k in ["prompt_tokens", "completion_tokens"]
            )
    
    def end_trace(self, final_output: str, success: bool = True, model: str = "gpt-4o"):
        if self.current_trace:
            self.current_trace.end_time = time.time()
            self.current_trace.final_output = final_output
            self.current_trace.success = success
            self.current_trace.total_cost_usd = self._calculate_cost(model)
            self._save_trace()
    
    def _calculate_cost(self, model: str) -> float:
        costs = self.MODEL_COSTS.get(model, {"input": 0.01, "output": 0.03})
        total_cost = 0.0
        for step in self.current_trace.steps:
            input_tokens = step.token_usage.get("prompt_tokens", 0)
            output_tokens = step.token_usage.get("completion_tokens", 0)
            total_cost += (input_tokens / 1000) * costs["input"]
            total_cost += (output_tokens / 1000) * costs["output"]
        return round(total_cost, 6)
    
    def _save_trace(self):
        filename = f"{self.log_dir}/trace_{self.current_trace.trace_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(asdict(self.current_trace), f, indent=2, default=str)
    
    def get_trace(self, trace_id: str) -> Optional[AgentTrace]:
        """Load a trace by ID."""
        import glob
        for path in glob.glob(f"{self.log_dir}/trace_{trace_id}_*.json"):
            with open(path) as f:
                data = json.load(f)
                return AgentTrace(**data)
        return None
    
    def replay_trace(self, trace_id: str, agent_func, **kwargs):
        """Re-run a trace through an agent for comparison."""
        trace = self.get_trace(trace_id)
        if not trace:
            return None
        
        print(f"Replaying trace {trace_id}")
        print(f"Original input: {trace.user_input}")
        print(f"Original output: {trace.final_output[:200]}...")
        print(f"Original cost: ${trace.total_cost_usd}")
        print()
        
        # Run the agent again with the same input
        result = agent_func(trace.user_input, **kwargs)
        print(f"New output: {result[:200]}...")
        
        return result
```
{% endraw %}

### Integrating the Logger into the ReAct Agent

{% raw %}
```python
class ObservableReActAgent:
    """ReAct agent with built-in observability."""
    
    def __init__(self, tools: list, model: str = "gpt-4o"):
        self.tools = tools
        self.model = model
        self.logger = AgentLogger()
        self.llm_client = openai.OpenAI()
    
    def run(self, user_input: str, session_id: str = None,
            max_steps: int = 15) -> str:
        
        session_id = session_id or str(uuid.uuid4())
        trace_id = self.logger.start_trace(session_id, user_input)
        
        # Build messages (same as ReAct agent from Post 1)
        messages = self._build_initial_messages(user_input)
        
        step_num = 0
        try:
            while step_num < max_steps:
                step_num += 1
                
                # LLM call with logging
                llm_start = time.time()
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0
                )
                llm_duration = time.time() - llm_start
                
                content = response.choices[0].message.content
                usage = response.usage.model_dump() if response.usage else {}
                
                # Log the LLM call step
                self.logger.log_step(AgentStep(
                    step_number=step_num,
                    agent_name="react_agent",
                    step_type="llm_call",
                    input_data={"messages": messages[-3:]},  # Last few messages
                    output_data=content,
                    token_usage=usage,
                    duration_ms=round(llm_duration * 1000, 2)
                ))
                
                # Check for final answer
                if "Final Answer:" in content:
                    final = content.split("Final Answer:")[-1].strip()
                    self.logger.end_trace(final, success=True, model=self.model)
                    return final
                
                # Parse and execute tool
                action_match = re.search(r"Action:\s*(\w+)", content)
                input_match = re.search(r"Action Input:\s*(\{.*\})", content, re.DOTALL)
                
                if action_match and input_match:
                    action_name = action_match.group(1)
                    action_input = json.loads(input_match.group(1))
                    
                    tool_start = time.time()
                    tool_result = self._execute_tool(action_name, action_input)
                    tool_duration = time.time() - tool_start
                    
                    # Log the tool call
                    self.logger.log_step(AgentStep(
                        step_number=step_num,
                        agent_name=action_name,
                        step_type="tool_call",
                        input_data=action_input,
                        output_data=tool_result[:500],  # Truncate for storage
                        duration_ms=round(tool_duration * 1000, 2)
                    ))
                    
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Observation: {tool_result}"})
            
            # Max steps reached
            self.logger.end_trace("Max steps reached", success=False, model=self.model)
            return "Max steps reached without a final answer."
            
        except Exception as e:
            self.logger.end_trace(str(e), success=False, model=self.model)
            raise
```
{% endraw %}

## Tracing with OpenTelemetry

For production systems, use [OpenTelemetry](https://opentelemetry.io/) — the industry standard for distributed tracing:

{% raw %}
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter, SimpleSpanProcessor
)
from opentelemetry.instrumentation.openai import OpenAIInstrumentor


# Set up tracing
provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Auto-instrument OpenAI calls
OpenAIInstrumentor().instrument()

# Create a tracer
tracer = trace.get_tracer("agent-tracer")


class TracedAgent:
    """Agent with OpenTelemetry spans."""
    
    def run(self, user_input: str) -> str:
        with tracer.start_as_current_span("agent_run") as span:
            span.set_attribute("user_input", user_input)
            
            with tracer.start_as_current_span("llm_reasoning") as llm_span:
                # LLM call happens here — automatically traced by OpenAIInstrumentor
                thought = self._reason(user_input)
                llm_span.set_attribute("thought", thought)
            
            with tracer.start_as_current_span("tool_execution") as tool_span:
                result = self._execute_tool(thought)
                tool_span.set_attribute("tool_result", result[:1000])
            
            span.set_attribute("final_result", result)
            return result
```
{% endraw %}

## Using LangFuse for Production Observability

[LangFuse](https://langfuse.com/) is an open-source observability platform purpose-built for LLM applications. It provides traces, cost tracking, and evaluation.

### Setup

```bash
pip install langfuse
```

{% raw %}
```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context


langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"  # Or self-hosted
)


class LangFuseAgent:
    """Agent instrumented with LangFuse observability."""
    
    @observe(name="agent_run", as_type="generation")
    def run(self, user_input: str) -> str:
        langfuse_context.update_current_observation(
            input=user_input,
            metadata={"agent_version": "2.1.0"}
        )
        
        result = self._agent_loop(user_input)
        
        langfuse_context.update_current_observation(
            output=result,
            usage={"input": 150, "output": 300}  # Token counts
        )
        
        return result
    
    @observe(name="tool_call")
    def _execute_tool(self, tool_name: str, **kwargs) -> str:
        langfuse_context.update_current_observation(
            input=kwargs,
            metadata={"tool": tool_name}
        )
        
        # Execute the tool
        result = self.tools[tool_name](**kwargs)
        
        langfuse_context.update_current_observation(
            output=result[:500]
        )
        
        return result
```
{% endraw %}

### What LangFuse Gives You

- **Trace view**: See every LLM call, tool execution, and decision in a timeline
- **Cost dashboard**: Per-session, per-user, per-model cost breakdowns
- **Latency analysis**: Identify slow steps in the agent loop
- **Session replay**: Re-watch an agent's exact trajectory
- **Evaluation**: Score agent outputs against expected results
- **Prompt management**: Version and A/B test prompts

## Agent Monitoring: Alerts and Anomalies

Beyond logging, production agents need **monitoring** — alerts when things go wrong:

{% raw %}
```python
import statistics


class AgentMonitor:
    """Monitor agent behavior and alert on anomalies."""
    
    def __init__(self, alert_webhook: str = None):
        self.metrics = {
            "latency_ms": [],
            "steps_per_session": [],
            "cost_per_session": [],
            "tool_error_rate": 0.0,
            "success_rate": 1.0
        }
        self.alert_webhook = alert_webhook  # Slack/PagerDuty webhook
        self.total_runs = 0
        self.failed_runs = 0
        self.tool_errors = 0
        self.total_tool_calls = 0
    
    def record_session(self, trace: AgentTrace):
        self.total_runs += 1
        if not trace.success:
            self.failed_runs += 1
        
        duration = trace.end_time - trace.start_time
        self.metrics["latency_ms"].append(duration * 1000)
        self.metrics["steps_per_session"].append(len(trace.steps))
        self.metrics["cost_per_session"].append(trace.total_cost_usd)
        
        for step in trace.steps:
            if step.step_type == "tool_call":
                self.total_tool_calls += 1
                if step.error:
                    self.tool_errors += 1
        
        self._check_alerts()
    
    def _check_alerts(self):
        alerts = []
        
        # Check success rate
        if self.total_runs >= 10:
            recent_failures = self.failed_runs / self.total_runs
            if recent_failures > 0.2:  # >20% failure rate
                alerts.append(f"High failure rate: {recent_failures:.1%}")
        
        # Check latency
        if len(self.metrics["latency_ms"]) >= 10:
            recent_latencies = self.metrics["latency_ms"][-10:]
            avg_latency = statistics.mean(recent_latencies)
            if avg_latency > 30000:  # >30 seconds
                alerts.append(f"High latency: {avg_latency:.0f}ms avg")
        
        # Check cost
        if len(self.metrics["cost_per_session"]) >= 10:
            recent_costs = self.metrics["cost_per_session"][-10:]
            avg_cost = statistics.mean(recent_costs)
            if avg_cost > 0.50:  # >$0.50 per session
                alerts.append(f"High cost: ${avg_cost:.3f} avg per session")
        
        # Check tool errors
        if self.total_tool_calls >= 20:
            error_rate = self.tool_errors / self.total_tool_calls
            if error_rate > 0.1:  # >10% tool error rate
                alerts.append(f"High tool error rate: {error_rate:.1%}")
        
        # Send alerts
        if alerts and self.alert_webhook:
            self._send_alert(alerts)
    
    def _send_alert(self, alerts: list[str]):
        import requests
        payload = {
            "text": f"Agent Monitor Alert:\n" + "\n".join(f"- {a}" for a in alerts)
        }
        try:
            requests.post(self.alert_webhook, json=payload, timeout=5)
        except:
            pass
```
{% endraw %}

## Practical Debugging Flow

When an agent produces a wrong answer, here's the debugging workflow:

1. **Check the trace**: What path did the agent take?
2. **Examine tool calls**: Did the tools return correct data?
3. **Look for prompt drift**: Did the LLM misinterpret instructions?
4. **Replay with fixes**: Modify the prompt/schema and re-run
5. **Compare traces**: Run the same input 3-5 times to see if the failure is consistent

## Dashboard Metrics to Track

| Metric | What It Tells You | Alert Threshold |
|--------|-------------------|-----------------|
| **Success rate** | How often agents complete tasks | < 80% |
| **Avg steps per session** | Agent efficiency | > 15 steps |
| **Avg latency per step** | Which steps are slow | > 10s per step |
| **Tool error rate** | Tool reliability issues | > 10% |
| **Cost per session** | Budget tracking | > $0.50 |
| **Re-query rate** | How often retries needed | > 20% |
| **Token waste** | Prompt optimization | > 30% context unused |

## Conclusion

Observability transforms agent development from guesswork into engineering. With structured logging, OpenTelemetry tracing, LangFuse dashboards, and automated monitoring, you can:

- **See exactly what your agent did** at every step
- **Know how much it cost** per session and per tool
- **Debug failures** by replaying traces with fixes
- **Get alerted** when things go wrong in production

**In the next post**, we'll take all of this and deploy agents to production — scaling, error handling, state persistence, and graceful degradation.

## Further Reading

- [LangFuse Documentation](https://langfuse.com/docs)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Multi-Agent Systems with LangGraph]({% post_url 2026-06-18-multi-agent-systems %})
- [Deploying AI Agents to Production]({% post_url 2026-06-20-agent-production-deployment %})
