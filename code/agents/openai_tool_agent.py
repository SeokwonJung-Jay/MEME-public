"""
OpenAI function-calling tool agent — reusable tool loop for file system operations.

Provides the same capabilities as Claude Agent SDK's Read/Write/Edit/Glob/Grep tools
but using OpenAI's function calling API.
"""

import hashlib
import json
import os
import glob as glob_mod
import re
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional


# ============================================================
# Optional tool-call JSONL logging (SIMBA uses this; main benchmark doesn't
# set the env var, so behavior is unchanged there).
# ============================================================

_TOOL_LOG_FH = None
_TOOL_LOG_LOCK = threading.Lock()


def _get_tool_log():
    global _TOOL_LOG_FH
    if _TOOL_LOG_FH is not None:
        return _TOOL_LOG_FH
    path = os.environ.get("TOOL_CALL_LOG_PATH")
    if not path:
        return None
    with _TOOL_LOG_LOCK:
        if _TOOL_LOG_FH is None:
            _TOOL_LOG_FH = open(path, "a", buffering=1)
    return _TOOL_LOG_FH


def _log_tool_call(tag: str, turn_idx: int, tc, response_meta: dict):
    fh = _get_tool_log()
    if fh is None:
        return
    args = tc.function.arguments
    rec = {
        "ts": time.time(),
        "thread": threading.current_thread().name,
        "tag": tag,
        "turn": turn_idx,
        "tool_name": tc.function.name,
        "args_len": len(args),
        "args_md5": hashlib.md5(args.encode()).hexdigest()[:10],
        "args": args,
        **response_meta,
    }
    with _TOOL_LOG_LOCK:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# Tool Definitions (OpenAI function calling schema)
# ============================================================

TOOL_READ = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file. Returns the file content as a string.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read (relative to working directory)"}
            },
            "required": ["path"]
        }
    }
}

TOOL_WRITE = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if needed. Overwrites existing content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write (relative to working directory)"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"]
        }
    }
}

TOOL_EDIT = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Replace a specific string in a file. The old_string must appear exactly once in the file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit"},
                "old_string": {"type": "string", "description": "Exact string to find and replace"},
                "new_string": {"type": "string", "description": "Replacement string"}
            },
            "required": ["path", "old_string", "new_string"]
        }
    }
}

TOOL_GLOB = {
    "type": "function",
    "function": {
        "name": "glob_files",
        "description": "Find files matching a glob pattern. Returns matching file paths.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.md', 'knowledge/concepts/*.md')"}
            },
            "required": ["pattern"]
        }
    }
}

TOOL_GREP = {
    "type": "function",
    "function": {
        "name": "grep_files",
        "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "File or directory to search in (default: current directory)"}
            },
            "required": ["pattern"]
        }
    }
}

ALL_TOOLS = [TOOL_READ, TOOL_WRITE, TOOL_EDIT, TOOL_GLOB, TOOL_GREP]
READ_ONLY_TOOLS = [TOOL_READ, TOOL_GLOB, TOOL_GREP]


# ============================================================
# Tool Execution
# ============================================================

def _resolve_path(cwd: Path, path_str: str) -> Path:
    """Resolve a path relative to cwd, ensuring it stays within cwd."""
    p = Path(path_str)
    if not p.is_absolute():
        p = cwd / p
    p = p.resolve()
    # Security: ensure path is within cwd
    if not str(p).startswith(str(cwd.resolve())):
        raise ValueError(f"Path {p} is outside working directory {cwd}")
    return p


def execute_tool(tool_name: str, args: Dict, cwd: Path) -> str:
    """Execute a file system tool and return the result string."""
    try:
        if tool_name == "read_file":
            p = _resolve_path(cwd, args["path"])
            if not p.exists():
                return f"Error: File '{args['path']}' not found."
            return p.read_text(encoding="utf-8")

        elif tool_name == "write_file":
            p = _resolve_path(cwd, args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"], encoding="utf-8")
            return f"OK: Written to '{args['path']}' ({len(args['content'])} chars)"

        elif tool_name == "edit_file":
            p = _resolve_path(cwd, args["path"])
            if not p.exists():
                return f"Error: File '{args['path']}' not found."
            content = p.read_text(encoding="utf-8")
            old = args["old_string"]
            new = args["new_string"]
            count = content.count(old)
            if count == 0:
                return f"Error: old_string not found in '{args['path']}'."
            if count > 1:
                return f"Error: old_string found {count} times in '{args['path']}'. Must be unique."
            content = content.replace(old, new, 1)
            p.write_text(content, encoding="utf-8")
            return f"OK: Edited '{args['path']}'"

        elif tool_name == "glob_files":
            pattern = args["pattern"]
            matches = sorted(glob_mod.glob(str(cwd / pattern), recursive=True))
            # Return relative paths
            rel = [str(Path(m).relative_to(cwd)) for m in matches if os.path.isfile(m)]
            if not rel:
                return "(no matches)"
            return "\n".join(rel)

        elif tool_name == "grep_files":
            pattern_str = args["pattern"]
            search_path = _resolve_path(cwd, args.get("path", "."))
            try:
                regex = re.compile(pattern_str)
            except re.error as e:
                return f"Error: Invalid regex: {e}"

            results = []
            if search_path.is_file():
                files = [search_path]
            else:
                files = sorted(search_path.rglob("*"))
                files = [f for f in files if f.is_file() and f.suffix in ('.md', '.txt', '.py', '.json', '.yaml', '.yml')]

            for fp in files[:50]:  # limit to 50 files
                try:
                    content = fp.read_text(encoding="utf-8")
                    for i, line in enumerate(content.splitlines(), 1):
                        if regex.search(line):
                            rel = fp.relative_to(cwd)
                            results.append(f"{rel}:{i}: {line}")
                except (UnicodeDecodeError, PermissionError):
                    continue

            if not results:
                return "(no matches)"
            return "\n".join(results[:100])  # limit output

        else:
            return f"Error: Unknown tool '{tool_name}'"

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ============================================================
# Tool Agent Loop
# ============================================================

def run_tool_agent(
    prompt: str,
    cwd: str | Path,
    model: str = "gpt-4.1-mini",
    system_prompt: str = None,
    tools: List[Dict] = None,
    max_turns: int = 30,
    api_key: str = None,
    log_tag: str = "",
) -> str:
    """Run an OpenAI function-calling agent with file system tools.

    Args:
        prompt: User prompt
        cwd: Working directory for file operations
        model: OpenAI model name
        system_prompt: Optional system prompt
        tools: Tool definitions (default: ALL_TOOLS)
        max_turns: Maximum conversation turns
        api_key: OpenAI API key (default: OPENAI_API_KEY env var)
        log_tag: Identifier for tool-call JSONL logging (no-op if env var unset)

    Returns:
        Final text response from the agent

    Env vars:
        OPENAI_SEED: integer passed as seed to OpenAI API (omitted if unset)
        TOOL_CALL_LOG_PATH: JSONL file to append tool-call records (off if unset)
    """
    # OpenRouter models (e.g., z-ai/glm-5.1) route via OpenRouter; plain
    # OpenAI models keep the default endpoint.
    from agents._model_utils import make_openai_client
    client = make_openai_client(model, api_key=api_key)
    cwd = Path(cwd).resolve()

    if tools is None:
        tools = ALL_TOOLS

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response_text = ""
    seed_env = os.environ.get("OPENAI_SEED")
    sys_md5 = hashlib.md5((system_prompt or "").encode()).hexdigest()[:10]

    from agents._model_utils import is_reasoning_model
    _is_reasoning = is_reasoning_model(model)

    for turn_idx in range(max_turns):
        if _is_reasoning:
            # GPT-5 / o-series ablation path.
            kwargs = dict(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                max_completion_tokens=4096,
                reasoning_effort="minimal",
            )
        else:
            # Main path (gpt-4.1-mini): untouched.
            kwargs = dict(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                temperature=0,
                max_tokens=4096,
            )
        if seed_env is not None:
            kwargs["seed"] = int(seed_env)
        response = client.chat.completions.create(**kwargs)

        msg = response.choices[0].message
        response_meta = {
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": response.choices[0].finish_reason,
            "sys_prompt_md5": sys_md5,
            "sys_prompt_len": len(system_prompt or ""),
        }

        if not msg.tool_calls:
            response_text = msg.content or ""
            break

        # Append assistant message with tool calls
        messages.append(msg)

        # Execute each tool call
        for tool_call in msg.tool_calls:
            _log_tool_call(log_tag, turn_idx, tool_call, response_meta)
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
                continue

            result = execute_tool(func_name, func_args, cwd)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return response_text
