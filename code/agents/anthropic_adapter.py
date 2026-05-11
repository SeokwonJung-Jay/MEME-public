"""
Anthropic → OpenAI compatibility adapter.

Makes the Anthropic API look like OpenAI's chat.completions.create()
so agents can use Claude models with zero code changes.

Usage:
    from agents.anthropic_adapter import AnthropicAsOpenAI
    client = AnthropicAsOpenAI(api_key=...)
    # Use exactly like OpenAI client:
    response = client.chat.completions.create(
        model="claude-sonnet-4-20250514",
        messages=[...],
        tools=[...],
        temperature=0,
        max_tokens=2000
    )
"""

import json
from anthropic import Anthropic


class _ToolCall:
    """Mimics openai.types.chat.ChatCompletionMessageToolCall."""
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _Function(name, arguments)
        self.type = "function"


class _Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _Message:
    """Mimics openai.types.chat.ChatCompletionMessage."""
    def __init__(self, content=None, tool_calls=None, role="assistant"):
        self.content = content
        self.tool_calls = tool_calls
        self.role = role

    def to_dict(self):
        """For appending to messages list (OpenAI pattern)."""
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in self.tool_calls
            ]
        return d

    # Make it dict-appendable to messages list
    def __getitem__(self, key):
        return self.to_dict()[key]

    def get(self, key, default=None):
        return self.to_dict().get(key, default)


class _Choice:
    def __init__(self, message):
        self.message = message


class _Response:
    def __init__(self, choices):
        self.choices = choices


class _Completions:
    """Mimics openai.resources.chat.Completions."""

    def __init__(self, anthropic_client):
        self._client = anthropic_client

    def create(self, model, messages, tools=None, temperature=0,
               max_tokens=2000, response_format=None, **kwargs):
        """Translate OpenAI-style call to Anthropic API."""

        # Convert messages: extract system, convert tool results
        system_text = ""
        anthropic_messages = []

        for msg in messages:
            role = msg["role"] if isinstance(msg, dict) else msg.role

            if role == "system":
                content = msg["content"] if isinstance(msg, dict) else msg.content
                system_text = content
                continue

            if role == "tool":
                # OpenAI tool result → Anthropic tool_result
                tool_call_id = msg["tool_call_id"] if isinstance(msg, dict) else msg.get("tool_call_id")
                content = msg["content"] if isinstance(msg, dict) else msg.get("content", "")
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content,
                        }
                    ]
                })
                continue

            if role == "assistant":
                # Could be a regular message or one with tool_calls
                if isinstance(msg, dict):
                    tool_calls = msg.get("tool_calls", [])
                    text_content = msg.get("content", "")
                else:
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    text_content = getattr(msg, "content", "") or ""

                if tool_calls:
                    content_blocks = []
                    if text_content:
                        content_blocks.append({"type": "text", "text": text_content})
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            tc_id = tc["id"]
                            tc_name = tc["function"]["name"]
                            tc_args = tc["function"]["arguments"]
                        else:
                            tc_id = tc.id
                            tc_name = tc.function.name
                            tc_args = tc.function.arguments

                        try:
                            input_obj = json.loads(tc_args)
                        except (json.JSONDecodeError, TypeError):
                            input_obj = {}

                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc_id,
                            "name": tc_name,
                            "input": input_obj,
                        })
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content_blocks,
                    })
                else:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": text_content or "",
                    })
                continue

            if role == "user":
                content = msg["content"] if isinstance(msg, dict) else msg.content
                anthropic_messages.append({
                    "role": "user",
                    "content": content,
                })
                continue

        # Merge consecutive user messages (Anthropic requires alternating roles)
        merged = []
        for msg in anthropic_messages:
            if merged and merged[-1]["role"] == msg["role"] == "user":
                # Merge content
                prev_content = merged[-1]["content"]
                new_content = msg["content"]
                if isinstance(prev_content, str) and isinstance(new_content, str):
                    merged[-1]["content"] = prev_content + "\n" + new_content
                elif isinstance(prev_content, list) and isinstance(new_content, list):
                    merged[-1]["content"] = prev_content + new_content
                elif isinstance(prev_content, str) and isinstance(new_content, list):
                    merged[-1]["content"] = [{"type": "text", "text": prev_content}] + new_content
                elif isinstance(prev_content, list) and isinstance(new_content, str):
                    merged[-1]["content"] = prev_content + [{"type": "text", "text": new_content}]
            else:
                merged.append(msg)
        anthropic_messages = merged

        # Convert OpenAI tool schemas to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = []
            for tool in tools:
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })

        # Build Anthropic API call.
        # Opus 4.7+ deprecates `temperature` and rejects requests that send it.
        api_kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        if "opus" not in model.lower():
            api_kwargs["temperature"] = temperature
        if system_text:
            api_kwargs["system"] = system_text
        if anthropic_tools:
            api_kwargs["tools"] = anthropic_tools

        # Call Anthropic API
        response = self._client.messages.create(**api_kwargs)

        # Convert response to OpenAI format
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(_ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input),
                ))

        content = "\n".join(text_parts) if text_parts else None
        message = _Message(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )

        return _Response(choices=[_Choice(message=message)])


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class AnthropicAsOpenAI:
    """Drop-in replacement for OpenAI() client that uses Anthropic API."""

    def __init__(self, api_key=None):
        self._anthropic = Anthropic(api_key=api_key)
        self.chat = _Chat(_Completions(self._anthropic))
