"""
MDFlatMemory — LLM agent that manages memory via markdown file tools.

The agent uses OpenAI function calling to read/write/append a memory.md file.
This is a direct evaluation of the "CLAUDE.md paradigm" — LLM writes its own
memory file and consults it when answering questions.
"""

import json
import os
from typing import List, Dict
from openai import OpenAI

from agents.base import BaseMemorySystem


# Process-global ep + session tracking; OpenRouter call patch reads these
# so any retry-log entry can be attributed to the right ep/session.
_CURRENT_EP_ID = None
_CURRENT_SESSION = None  # (session_id, type)


def _log_glm_event(kind, exc, content_preview=None):
    """Per-worker GLM event log. Each worker writes to pid-tagged file
    so output isn't interleaved or buffered-lost via tee.
    `kind` ∈ {'empty_response', 'no_tool_calls', 'parse_error'}."""
    try:
        ep = _CURRENT_EP_ID or 'unknown'
        sess = _CURRENT_SESSION or ('?', '?')
        msg = f'{type(exc).__name__}: {str(exc)[:200]}' if exc else (content_preview or '')
        line = (f'[pid={os.getpid()} ep={ep} session={sess[0]} type={sess[1]} '
                f'{kind}] {msg}\n')
        path = os.environ.get('GLM_RETRY_LOG', f'/tmp/glm_retry.pid{os.getpid()}.log')
        with open(path, 'a') as f:
            f.write(line)
    except Exception:
        pass


# ============================================================
# File System Tools (OpenAI function calling schema)
# ============================================================

MEMORY_FILE = "memory.md"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Read the current contents of your memory file.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_memory",
            "description": "Overwrite the entire memory file with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "append_memory",
            "description": "Append content to the end of your memory file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to append (will be added on a new line)"}
                },
                "required": ["content"]
            }
        }
    },
]

# Read-only for retrieve (prevent memory pollution during question answering)
RETRIEVE_TOOLS = [t for t in TOOLS if t["function"]["name"] == "read_memory"]


# ============================================================
# Virtual File System
# ============================================================

class VirtualFileSystem:
    """In-memory virtual file system for tool execution."""

    def __init__(self):
        self.files = {}

    def read_file(self, path: str) -> str:
        if path in self.files:
            return self.files[path]
        return f"Error: File '{path}' not found."

    def write_file(self, path: str, content: str) -> str:
        self.files[path] = content
        return f"OK: Written to '{path}' ({len(content)} chars)"

    def append_file(self, path: str, content: str) -> str:
        if path in self.files:
            self.files[path] += "\n" + content
        else:
            self.files[path] = content
        return f"OK: Appended to '{path}'"

    def list_files(self) -> str:
        if not self.files:
            return "(no files)"
        return "\n".join(self.files.keys())

    def execute_tool(self, tool_name: str, args: Dict) -> str:
        REQUIRED_ARGS = {
            "read_memory": [],
            "write_memory": ["content"],
            "append_memory": ["content"],
        }

        if tool_name not in REQUIRED_ARGS:
            return f"Error: Unknown tool '{tool_name}'"

        missing = [k for k in REQUIRED_ARGS[tool_name] if k not in args]
        if missing:
            return f"Error: Missing required arguments {missing} for {tool_name}. Got: {list(args.keys())}"

        if tool_name == "read_memory":
            return self.read_file(MEMORY_FILE)
        elif tool_name == "write_memory":
            return self.write_file(MEMORY_FILE, args["content"])
        elif tool_name == "append_memory":
            return self.append_file(MEMORY_FILE, args["content"])

    def get_all(self) -> str:
        if not self.files:
            return "(empty)"
        parts = []
        for path, content in sorted(self.files.items()):
            parts.append(f"=== {path} ===\n{content}")
        return "\n\n".join(parts)

    def reset(self):
        self.files = {}


# ============================================================
# Prompts
# ============================================================

INGEST_PROMPT = """You are a personal assistant with a persistent memory file (memory.md).
After each conversation, save any information the user shared that may be useful in future sessions.
Keep it compact — one fact per line with timestamp [YYYY/MM/DD].

If information has changed, update it. If something was removed or cancelled, remove the old entry.

Do NOT save conversation summaries, assistant responses, or temporary task state.

You will now receive a conversation session. Read it and update your memory accordingly."""


MAX_TOOL_ROUNDS = 5


# ============================================================
# MDFlatMemory
# ============================================================

class MDFlatMemory(BaseMemorySystem):
    """LLM agent that manages memory via markdown file tools (OpenAI function calling)."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str = None, client=None,
                 ingest_prompt=None, internal_model=None, **kwargs):
        self.model = model
        self.client = client or (OpenAI(api_key=api_key) if api_key else OpenAI())

        # Internal LLM for tool-calling loops (ingest + retrieve).
        # Default: same as the answering model. Override with --internal-model.
        # The answering-LLM api_key (passed in `api_key`) is NEVER reused for
        # the internal client: when internal and answering providers differ
        # (e.g., gpt-4.1-mini answer + Claude internal), reusing the answering
        # key would 401. Read each provider's key directly from env instead.
        im = internal_model or model
        if im.startswith("claude"):
            from agents.anthropic_adapter import AnthropicAsOpenAI
            self._internal_client = AnthropicAsOpenAI(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        else:
            # OpenRouter models (e.g., z-ai/glm-5.1) route via OpenRouter;
            # plain OpenAI models keep the default endpoint.
            from agents._model_utils import make_openai_client
            self._internal_client = make_openai_client(im)
        self._internal_model = im

        self.fs = VirtualFileSystem()
        self._last_retrieved_context = ""
        self.ingest_prompt = ingest_prompt or INGEST_PROMPT

    def _run_tool_loop(self, messages: List[Dict], tools=None) -> Dict:
        """Tool calling loop. Returns {"response": str, "trajectory": list, "token_usage": dict}."""
        if tools is None:
            tools = TOOLS
        trajectory = []
        token_usage = {"input_tokens": 0, "output_tokens": 0}

        from agents._model_utils import is_reasoning_model, is_openrouter_model
        _is_reasoning = is_reasoning_model(self._internal_model)
        _is_or = is_openrouter_model(self._internal_model)

        for _ in range(MAX_TOOL_ROUNDS):
            if _is_reasoning:
                # GPT-5 / o-series ablation path: reasoning models reject
                # temperature + max_tokens, require max_completion_tokens.
                kw = dict(
                    model=self._internal_model,
                    messages=messages,
                    tools=tools,
                    max_completion_tokens=2000,
                )
                if _is_or:
                    # GLM via OpenRouter: reasoning must be disabled (any
                    # effort returns content=None on complex prompts).
                    # Novita excluded for additional stability.
                    kw['extra_body'] = {
                        'reasoning': {'enabled': False},
                        'provider': {'ignore': ['novita']},
                    }
                else:
                    # OpenAI reasoning models (gpt-5/o-series): pass
                    # reasoning_effort kwarg directly (works correctly).
                    kw['reasoning_effort'] = "minimal"
                response = self._internal_client.chat.completions.create(**kw)
            else:
                # Main path (gpt-4.1-mini): untouched.
                response = self._internal_client.chat.completions.create(
                    model=self._internal_model,
                    messages=messages,
                    tools=tools,
                    temperature=0,
                    max_tokens=2000
                )

            # Accumulate token usage
            if hasattr(response, 'usage') and response.usage:
                token_usage["input_tokens"] += getattr(response.usage, 'prompt_tokens', 0) or getattr(response.usage, 'input_tokens', 0) or 0
                token_usage["output_tokens"] += getattr(response.usage, 'completion_tokens', 0) or getattr(response.usage, 'output_tokens', 0) or 0

            msg = response.choices[0].message

            if not msg.tool_calls:
                # Detect potential GLM/OpenRouter issue: no tool_calls AND
                # empty content == model produced nothing useful.
                if _is_or and not msg.content:
                    _log_glm_event('empty_response', None, content_preview='no_tool_calls + content=None')
                return {"response": msg.content or "", "trajectory": trajectory, "token_usage": token_usage}

            messages.append(msg)

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    result = f"Error: Could not parse arguments for {func_name}."
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                    trajectory.append({"tool": func_name, "args": None, "result": result})
                    continue

                result = self.fs.execute_tool(func_name, func_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                trajectory.append({"tool": func_name, "args": func_args, "result": result})

        return {"response": "", "trajectory": trajectory, "token_usage": token_usage}

    def _set_user_id(self, user_id):
        """Track ep namespace for retry-log attribution. md_file itself
        doesn't use namespaces (in-process VFS) but we record it for logging."""
        global _CURRENT_EP_ID
        _CURRENT_EP_ID = str(user_id)

    def ingest_session(self, session: dict) -> dict:
        """Process a session and update memory.md via tool calls."""
        # Track current session so retry-log entries can be attributed to
        # the specific session (filler vs evidence) that triggered them.
        global _CURRENT_SESSION
        _CURRENT_SESSION = (session.get('session_id', '?'), session.get('type', '?'))

        conv_text = f"[Session: {session.get('timestamp', 'unknown')}]\n"
        for turn in session["conversation"]:
            role = "User" if turn['role'] == 'user' else "Assistant"
            conv_text += f"{role}: {turn['content']}\n"

        messages = [
            {"role": "system", "content": self.ingest_prompt},
            {"role": "user", "content": conv_text}
        ]
        result = self._run_tool_loop(messages)
        return {"trajectory": result["trajectory"], "token_usage": result["token_usage"]}

    def retrieve(self, question: str) -> str:
        """MD-flat native retrieval: agent reads memory.md via tool call and extracts relevant facts."""
        retrieve_prompt = (
            f"Extract ONLY the facts relevant to answering the question below. "
            f"Read your memory file first, then return relevant facts as-is "
            f"(do not rephrase or summarize). "
            f"If nothing is relevant, say '(no relevant facts)'.\n\n"
            f"Question: {question}"
        )
        messages = [
            {"role": "system", "content": "You have access to read_memory(). Read your memory to find relevant facts."},
            {"role": "user", "content": retrieve_prompt},
        ]
        result = self._run_tool_loop(messages, tools=RETRIEVE_TOOLS)
        # Accumulate retrieve token usage
        if not hasattr(self, '_retrieve_token_usage'):
            self._retrieve_token_usage = {"input_tokens": 0, "output_tokens": 0}
        ru = result.get("token_usage", {})
        self._retrieve_token_usage["input_tokens"] += ru.get("input_tokens", 0)
        self._retrieve_token_usage["output_tokens"] += ru.get("output_tokens", 0)
        return result["response"] if result["response"] else "(no relevant facts)"

    def get_memory_snapshot(self) -> dict:
        """Return full memory file contents."""
        text = self.fs.get_all()
        return {"type": "md_file", "text": text}

    def reset(self):
        """Clear all files."""
        self.fs.reset()
        self._last_retrieved_context = ""
