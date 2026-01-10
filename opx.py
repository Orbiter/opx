#!/usr/bin/env python3
import os
import re
import sys
import json
import shlex
import argparse
import subprocess
import time
import io
import textwrap
import http.client
import urllib.error
import urllib.request
from html.parser import HTMLParser

USAGE = """Usage: opx.py [options] <prompt>
Options:
  -m <model>      model name
  -h <host>       hostname
  -p <port>       port number
  -e <file>       read file content instead of stdin
  --help          show help and exit
"""

SYSTEM_PROMPT = (
    "You are a mighty Linux system operator. Make short answers. Use tools as many times as you require to solve the given task. Don't use '|', ';', '&', '>' or '<' in bash commands."
)

DEFAULT_MODEL = "qwen3-vl:4b-instruct-q4_K_M"
#DEFAULT_MODEL = "devstral-small-2:24b-instruct-2512-q4_K_M"
#DEFAULT_MODEL = "hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_M"

def usage(exit_code=0, err=False):
    if err:
        termprint("error", USAGE)
    else:
        termprint("paragraph", USAGE.rstrip("\n"))
    sys.exit(exit_code)

def parse_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-m", dest="model", default=DEFAULT_MODEL)
    parser.add_argument("-h", dest="host", default="localhost")
    parser.add_argument("-p", dest="port", default="11434")
    parser.add_argument("-e", dest="input_file", type=argparse.FileType("r", encoding="utf-8"))
    parser.add_argument("--help", action="store_true")
    parser.add_argument("prompt", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.help: usage(0)
    prompt_parts = args.prompt
    if prompt_parts and prompt_parts[0] == "--": prompt_parts = prompt_parts[1:]
    if not prompt_parts: usage(1, True)
    #if not prompt_parts: prompt_parts = ["show me the directory listing of this folder"] #debugging

    prompt = " ".join(prompt_parts)
    return args.host, args.port, args.model, args.input_file, prompt

def _http_request(host, port, method, path, body=None, read_body=True, leave_open=False):
    conn = http.client.HTTPConnection(host, int(port), timeout=60)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        if resp is None or resp.status < 200 or resp.status >= 300:
            if leave_open and resp: resp.close()
            return None, None
        if not read_body: return resp, None
        return resp, resp.read()
    except (OSError, http.client.HTTPException):
        return None, None
    finally:
        if not leave_open:
            conn.close()

def classifier(classes_description, classes, proposition, compute_confidence=True, binary=True, host="localhost", port="11434", model=DEFAULT_MODEL):
    schema = {
        "title": "Classifier",
        "type": "object",
        "properties": {"classification": {"type": "literal", "enum": classes}},
        "required": ["classification"],
    }
    if compute_confidence:
        schema["properties"]["confidence"] = {"type": "number", "minimum": 0, "maximum": 100}
        schema["required"].append("confidence")
    messages = [
        {"role": "system", "content": classes_description},
        {"role": "user", "content": proposition},
    ]
    data = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 2048,
        "messages": messages,
        "stream": False,
        "response_format": {"type": "json_schema", "json_schema": {"strict": True, "schema": schema}},
    }
    resp, raw = _http_request(host, port, "POST", "/v1/chat/completions", json.dumps(data))
    if not resp or not raw: return "", 0.0
    try:
        response = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return "", 0.0
    content_text = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    try:
        content = json.loads(content_text)
    except json.JSONDecodeError:
        content = {}
    classification = content.get("classification", "")
    if classification.startswith("'"): classification = classification[1:]
    if classification.endswith("'"): classification = classification[:-1]
    confidence = content.get("confidence", 0.5) if compute_confidence else 1.0
    if confidence > 1: confidence = confidence / 100
    return classification, confidence

def truth_test(classes_description, proposition, compute_confidence=True, host="localhost", port="11434", model=DEFAULT_MODEL):
    classification, confidence = classifier(classes_description, ["true", "false"], proposition, compute_confidence=compute_confidence, binary=True, host=host, port=port, model=model)
    return classification == "true", confidence

def _format_conversation(messages):
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        if role == "assistant" and "tool_calls" in msg:
            parts.append(f"assistant tool_calls: {json.dumps(msg.get('tool_calls'))}")
            continue
        content = msg.get("content")
        if content is not None: parts.append(f"{role}: {content}")
    return "\n\n".join(parts)

def _build_follow_up_prompt(conversation, initial_prompt, host, port, model):
    system_prompt = (
        "You create follow-up prompts for a model to complete the original request."
        " Return only the follow-up prompt, no extra text."
    )
    user_prompt = (
        "Initial request:\n"
        f"{initial_prompt}\n\n"
        "Conversation so far:\n"
        f"{conversation}\n\n"
        "Review the answer and construct a prompt that uses the activities and their effect "
        "so far to define a new prompt which shall work toward a solution of the initial prompt."
    )
    data = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    resp, raw = _http_request(host, port, "POST", "/v1/chat/completions", json.dumps(data))
    if not resp or not raw:
        return ""
    try:
        response = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return ""
    content_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content_text.strip()

def ensure_model_available(host, port, model):
    resp, data = _http_request(host, port, "GET", "/api/tags")
    if not resp or not data: return
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return
    models = payload.get("models") or []
    if any(m.get("name") == model for m in models):
        return
    pull_body = json.dumps({"name": model, "stream": False})
    resp, _ = _http_request(host, port, "POST", "/api/pull", pull_body)
    if not resp:
        termprint("error", f"Failed to pull model: {model}\n")
        sys.exit(1)

approval_always_read = False
approval_always_write = False

def request_tool_approval(subject, write_request):
    global approval_always_read, approval_always_write
    if write_request and approval_always_write:
        return True
    if not write_request and approval_always_read:
        return True
    default_prompt = "[y/N]" if write_request else "[Y/n]"
    subject_line = " ".join(str(subject).split())
    if len(subject_line) > 120: subject_line = subject_line[:117] + "..."
    termprint("paragraph", f"Approve '{subject_line}', {default_prompt} (a=always):")
    resp = sys.stdin.readline().strip().lower()
    if resp == "a":
        if write_request:
            approval_always_write = True
        else:
            approval_always_read = True
        return True
    if (not resp): return not write_request # default value from request options
    return resp in ["Y", "y", "yes"]

def _load_tool_args(arguments):
    if not arguments:
        return None
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return None

def _path_tool_description(name, description):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to target.",
                    }
                },
                "required": ["path"], "additionalProperties": False,
            },
            "strict": True,
        },
    }

class BaseTool:
    name = ""

    def describe(self):
        raise NotImplementedError

    def argument_spec(self):
        return {}

    def parse_arguments(self, arguments):
        args = _load_tool_args(arguments)
        if not isinstance(args, dict):
            return None
        spec = self.argument_spec()
        if not spec:
            return {}
        parsed = {}
        for key, expected_type in spec.items():
            value = args.get(key)
            if expected_type is str:
                if not isinstance(value, str):
                    return None
            elif not isinstance(value, expected_type):
                return None
            parsed[key] = value
        return parsed

    def handle(self, **kwargs):
        raise NotImplementedError

    def handle_request(self, tool_request):
        return self.handle(**tool_request)

class BashTool(BaseTool):
    name = "bash"

    def argument_spec(self):
        return {"command": str}

    def describe(self):
        return {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a single shell command and return stdout/stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Single shell command (no pipes or redirection).",
                        }
                    },
                    "required": ["command"], "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def handle(self, command=None, **_):
        if not command:
            return {"tool": "bash", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Empty command", "data": {}, "message": "Empty command"}
        forbidden = ["|", ";", "&", ">", "<", "\n", "\r"]
        if any(token in command for token in forbidden):
            reject_msg = "Rejected: unsafe command; don't use '|', ';', '&', '>', '<'; try a safe approach"
            return {"tool": "bash", "ok": False, "exit_code": 1, "stdout": "", "stderr": reject_msg, "data": {}, "message": reject_msg}
        if not request_tool_approval(f"bash: {command}", write_request=True):
            reject_msg = "Rejected by user, try a different approach"
            return {"tool": "bash", "ok": False, "exit_code": 1, "stdout": "", "stderr": reject_msg, "data": {}, "message": reject_msg}
        try:
            args = shlex.split(command)
        except ValueError as exc:
            return {"tool": "bash", "ok": False, "exit_code": 1, "stdout": "", "stderr": str(exc), "data": {}, "message": str(exc)}
        if not args:
            return {"tool": "bash", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Empty command", "data": {}, "message": "Empty command"}
        try:
            completed = subprocess.run(args, capture_output=True, text=True)
        except OSError as exc:
            return {"tool": "bash", "ok": False, "exit_code": 1, "stdout": "", "stderr": str(exc), "data": {}, "message": str(exc)}
        return {"tool": "bash", "ok": completed.returncode == 0, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "data": {"command": command, "args": args}}

class EditTool(BaseTool):
    name = "edit"

    def argument_spec(self):
        return {"diff": str}

    def describe(self):
        return {
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Apply a unified diff to edit files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "diff": {
                            "type": "string",
                            "description": "Unified diff of the file edits to apply.",
                        }
                    },
                    "required": ["diff"], "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def _count_files_changed(self, diff_body):
        files = set()
        for line in diff_body.splitlines():
            if line.startswith("+++ "):
                path = line[4:].strip()
                if path and path != "/dev/null":
                    files.add(path)
        return len(files)

    def handle(self, diff=None, **_):
        if not diff:
            return {"tool": "edit", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Empty diff", "data": {"applied": False, "files_changed": 0}, "message": "Empty diff"}
        termprint("subitem", "Tool request (edit), diff:")
        termprint("box", diff)
        if not request_tool_approval(diff, write_request=True):
            return {"tool": "edit", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Rejected by user", "data": {"applied": False, "files_changed": 0}, "message": "Rejected by user"}
        try:
            completed = subprocess.run(
                ["patch", "-p0", "--forward"],
                input=diff,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            return {"tool": "edit", "ok": False, "exit_code": 1, "stdout": "", "stderr": str(exc), "data": {"applied": False, "files_changed": 0}, "message": str(exc)}
        files_changed = self._count_files_changed(diff)
        return {"tool": "edit", "ok": completed.returncode == 0, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "data": {"applied": completed.returncode == 0, "files_changed": files_changed}}

class WriteTool(BaseTool):
    name = "write"

    def argument_spec(self):
        return {"path": str, "content": str}

    def describe(self):
        return {
            "type": "function",
            "function": {
                "name": "write",
                "description": "Create a new file with provided content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the new file (must not already exist).",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full content of the new file.",
                        },
                    },
                    "required": ["path", "content"], "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def handle(self, path=None, content=None, **_):
        if not path:
            return {"tool": "write", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Missing path", "data": {}, "message": "Missing path"}
        if content is None:
            return {"tool": "write", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Missing content", "data": {}, "message": "Missing content"}
        termprint("subitem", f"Tool request (write), path: {path}")
        termprint("box", content)
        if not request_tool_approval(f"write to: {path}", write_request=True):
            return {"tool": "write", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Rejected by user", "data": {}, "message": "Rejected by user"}
        try:
            with open(path, "x", encoding="utf-8") as f:
                f.write(content)
        except FileExistsError:
            return {"tool": "write", "ok": False, "exit_code": 1, "stdout": "", "stderr": "File already exists", "data": {}, "message": "File already exists"}
        except OSError as exc:
            return {"tool": "write", "ok": False, "exit_code": 1, "stdout": "", "stderr": str(exc), "data": {}, "message": str(exc)}
        return {"tool": "write", "ok": True, "exit_code": 0, "stdout": "OK\n", "stderr": "", "data": {"path": path, "bytes": len(content.encode("utf-8"))}}

class ReadTool(BaseTool):
    name = "read"

    def argument_spec(self):
        return {"path": str}

    def describe(self):
        return _path_tool_description(
            "read",
            "Read a text file and return its contents.",
        )

    def handle(self, path=None, **_):
        if not path: return {"tool": "read", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Missing path", "data": {}, "message": "Missing path"}
        termprint("subitem", f"Tool request (read), path: {path}")
        if not request_tool_approval(f"read from: {path}", write_request=False):
            return {"tool": "read", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Rejected by user", "data": {}, "message": "Rejected by user"}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
        except OSError as exc:
            return {"tool": "read", "ok": False, "exit_code": 1, "stdout": "", "stderr": str(exc), "data": {}, "message": str(exc)}
        return {"tool": "read", "ok": True, "exit_code": 0, "stdout": data, "stderr": "", "data": {"path": path, "content": data, "bytes": len(data.encode("utf-8"))}}

class ListTool(BaseTool):
    name = "list"

    def argument_spec(self):
        return {"path": str}

    def describe(self):
        return _path_tool_description(
            "list",
            "List directory entries.",
        )

    def handle(self, path=None, **_):
        if not path: return {"tool": "list", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Missing path", "data": {}, "message": "Missing path"}
        termprint("subitem", f"Tool request (list), path: {path}")
        if not request_tool_approval(f"list: {path}", write_request=False):
            return {"tool": "list", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Rejected by user", "data": {}, "message": "Rejected by user"}
        try:
            entries = os.listdir(path)
        except OSError as exc:
            return {"tool": "list", "ok": False, "exit_code": 1, "stdout": "", "stderr": str(exc), "data": {}, "message": str(exc)}
        entries.sort()
        return {"tool": "list", "ok": True, "exit_code": 0, "stdout": "\n".join(entries) + "\n", "stderr": "", "data": {"files": entries, "count": len(entries)}}

class MkdirTool(BaseTool):
    name = "mkdir"

    def argument_spec(self):
        return {"path": str}

    def describe(self):
        return _path_tool_description(
            "mkdir",
            "Create a new directory.",
        )

    def handle(self, path=None, **_):
        if not path: return {"tool": "mkdir", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Missing path", "data": {}, "message": "Missing path"}
        termprint("subitem", f"Tool request (mkdir), path: {path}")
        if not request_tool_approval(f"mkdir: {path}", write_request=True):
            return {"tool": "mkdir", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Rejected by user", "data": {}, "message": "Rejected by user"}
        try:
            os.makedirs(path, exist_ok=False)
        except FileExistsError:
            return {"tool": "mkdir", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Already exists", "data": {}, "message": "Already exists"}
        except OSError as exc:
            return {"tool": "mkdir", "ok": False, "exit_code": 1, "stdout": "", "stderr": str(exc), "data": {}, "message": str(exc)}
        return {"tool": "mkdir", "ok": True, "exit_code": 0, "stdout": "OK", "stderr": "", "data": {"path": path}}

class HTMLToMarkdown(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []
        self.in_pre = False
        self.link_href = None
        self.link_text = []

    def _ensure_blankline(self):
        if not self.out:
            return
        text = "".join(self.out)
        if not text.endswith("\n\n"):
            if text.endswith("\n"):
                self.out.append("\n")
            else:
                self.out.append("\n\n")

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self._ensure_blankline()
        elif tag == "br":
            self.out.append("\n")
        elif tag in ["strong", "b"]:
            self.out.append("**")
        elif tag in ["em", "i"]:
            self.out.append("*")
        elif tag == "code":
            if not self.in_pre: self.out.append("`")
        elif tag == "pre":
            self._ensure_blankline()
            self.out.append("```\n")
            self.in_pre = True
        elif tag == "a":
            self.link_href = None
            self.link_text = []
            for key, value in attrs:
                if key == "href": self.link_href = value

    def handle_endtag(self, tag):
        if tag in ["strong", "b"]:
            self.out.append("**")
        elif tag in ["em", "i"]:
            self.out.append("*")
        elif tag == "code":
            if not self.in_pre: self.out.append("`")
        elif tag == "pre":
            if not "".join(self.out).endswith("\n"): self.out.append("\n")
            self.out.append("```\n")
            self.in_pre = False
        elif tag == "a":
            text = "".join(self.link_text).strip()
            href = self.link_href or ""
            if text: self.out.append(f"[{text}]({href})")
            self.link_href = None
            self.link_text = []

    def handle_data(self, data):
        if not data: return
        if self.in_pre:
            self.out.append(data)
            return
        if self.link_href is not None:
            self.link_text.append(data)
        else:
            self.out.append(re.sub(r"\s+", " ", data))

    def get_markdown(self):
        return "".join(self.out).strip() + "\n"

def _parse_charset(content_type):
    if not content_type: return None
    match = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
    if match: return match.group(1).strip()
    return None

def _is_text_mime(content_type):
    if not content_type: return False
    mime = content_type.split(";", 1)[0].strip().lower()
    return mime.startswith("text/")

class InternetReadTool(BaseTool):
    name = "internet_read"

    def argument_spec(self):
        return {"url": str}

    def describe(self):
        return {
            "type": "function",
            "function": {
                "name": "internet_read",
                "description": "Read a text resource from a URL; HTML is converted to Markdown.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch (text/* only).",
                        }
                    },
                    "required": ["url"], "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def handle(self, url=None, **_):
        if not url: return {"tool": "internet_read", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Missing url", "data": {}, "message": "Missing url"}
        termprint("subitem", f"Tool request (internet_read), url: {url}")
        if not request_tool_approval(url, write_request=False):
            return {"tool": "internet_read", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Rejected by user", "data": {}, "message": "Rejected by user"}
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "opx/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if not _is_text_mime(content_type):
                    return {"tool": "internet_read", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Rejected: non-text mime type", "data": {}, "message": "Rejected: non-text mime type"}
                raw = resp.read()
        except urllib.error.URLError as exc:
            return {"tool": "internet_read", "ok": False, "exit_code": 1, "stdout": "", "stderr": str(exc), "data": {}, "message": str(exc)}
        charset = _parse_charset(content_type) or "utf-8"
        text = raw.decode(charset, errors="replace")
        if content_type.lower().startswith("text/html"):
            parser = HTMLToMarkdown()
            parser.feed(text)
            text = parser.get_markdown()
        return {"tool": "internet_read", "ok": True, "exit_code": 0, "stdout": text, "stderr": "", "data": {"url": url, "content": text, "bytes": len(text.encode("utf-8"))}}

def _tool_instances():
    return [
        BashTool(),
        EditTool(),
        WriteTool(),
        ReadTool(),
        ListTool(),
        MkdirTool(),
        InternetReadTool(),
    ]

TOOL_INSTANCES = _tool_instances()
TOOL_REGISTRY = {tool.name: tool for tool in TOOL_INSTANCES}

DEFAULT_STREAM_PREFIX = ">>> "

class _LinePrefixWriter:
    def __init__(self, writer, prefix=DEFAULT_STREAM_PREFIX, max_line_length=132):
        self.writer = writer
        self.prefix = prefix
        self.at_line_start = True
        self.line_length = 0
        self.max_line_length = max_line_length

    def write(self, content):
        if not content or self.writer is None:
            return
        pending_word = ""

        def flush_word():
            nonlocal pending_word
            if not pending_word:
                return
            word_len = len(pending_word)
            if self.at_line_start:
                self.writer.write(self.prefix)
                self.at_line_start = False
            if self.line_length > 0 and self.line_length + word_len > self.max_line_length:
                self.writer.write("\n")
                self.writer.write(self.prefix)
                self.line_length = 0
            self.writer.write(pending_word)
            self.line_length += word_len
            pending_word = ""

        for ch in content:
            if ch == "\n":
                flush_word()
                self.writer.write(ch)
                self.at_line_start = True
                self.line_length = 0
                continue
            if ch == " ":
                flush_word()
                if self.line_length > 0 and self.line_length + 1 > self.max_line_length:
                    self.writer.write("\n")
                    self.at_line_start = True
                    self.line_length = 0
                if not self.at_line_start:
                    self.writer.write(" ")
                    self.line_length += 1
                continue
            pending_word += ch

        flush_word()

    def flush(self):
        if self.writer:
            self.writer.flush()

def _emit_content(content, writer):
    if not content: return
    if writer is None: return
    writer.write(content)
    writer.flush()

def process_response(resp, writer, tool_registry=None, content_collector=None):
    tool_calls_data = {}
    tool_call_order = []
    if tool_registry is None:
        tool_registry = TOOL_REGISTRY

    def _add_or_update_tool_call(call):
        key = call.get("id")
        if key is None and call.get("index") is not None:
            key = f"index_{call.get('index')}"
        if key is None:
            key = f"call_{len(tool_call_order) + 1}"
        if key not in tool_calls_data:
            tool_calls_data[key] = {"id": call.get("id"), "name": None, "arguments": ""}
            tool_call_order.append(key)
        func = call.get("function") or {}
        name = func.get("name")
        arguments = func.get("arguments")
        if name:
            tool_calls_data[key]["name"] = name
        if arguments:
            tool_calls_data[key]["arguments"] += arguments

    def _build_tool_request(tool_call):
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments") or ""
        tool_id = tool_call.get("id")
        if tool_args and tool_name is None:
            tool_name = "bash"
        tool = tool_registry.get(tool_name)
        if tool and tool_args:
            parsed_args = tool.parse_arguments(tool_args)
            if parsed_args is not None:
                tool_request = {"id": tool_id, "name": tool_name, "arguments": tool_args}
                tool_request.update(parsed_args)
                return tool_request
        return None

    while True:
        line = resp.readline()
        if not line: break
        if not line.startswith(b"data:"): continue
        data = line[5:].strip()
        if data == b"[DONE]": break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = payload.get("choices") or []
        if not choices: continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        tool_calls = delta.get("tool_calls") or []
        if content_collector is not None and content: content_collector.append(content)
        if not tool_calls:
            _emit_content(content, writer)
        for call in tool_calls:
            _add_or_update_tool_call(call)

    if writer: writer.flush()
    tool_requests = []
    for key in tool_call_order:
        tool_request = _build_tool_request(tool_calls_data.get(key, {}))
        if tool_request: tool_requests.append(tool_request)
    return tool_requests

def _format_title(base, elapsed):
    if elapsed is None: return base
    return f"{base} ({elapsed:.2f}s)"

def termprint(style: str, content: str, title=None, elapsed=None, writer=sys.stdout, width=132):
    header = _format_title(title, elapsed) if title else ""
    text = content or ""
    if header:
        text = f"{header}\n{text}" if text else header
    if style == "paragraph":
        writer.write(text)
        if text and not text.endswith("\n"): writer.write("\n")
        writer.flush()
        return
    if style == "item":
        wrap_width = max(1, width - 2)
        filled = textwrap.fill( text, width=wrap_width, initial_indent="* ", subsequent_indent="  ")
        writer.write(filled + "\n")
        writer.flush()
        return
    if style == "subitem":
        wrap_width = max(1, width - 4)
        filled = textwrap.fill( text, width=wrap_width, initial_indent="  └ ", subsequent_indent="    ")
        writer.write(filled + "\n")
        writer.flush()
        return
    if style == "error":
        wrap_width = max(1, width - 6)
        filled = textwrap.fill( text, width=wrap_width, initial_indent="ERROR ", subsequent_indent="      ")
        writer.write(filled + "\n")
        writer.flush()
        return
    if style == "box":
        lines = text.splitlines() if text else [""]
        top = "╭" + ("─" * width) + "╮"
        bottom = "╰" + ("─" * width) + "╯"
        writer.write(top + "\n")
        for line in lines:
            line = line.expandtabs(4)
            clipped = line[:width]
            writer.write("│" + clipped.ljust(width) + "│\n")
        writer.write(bottom + "\n")
        writer.flush()
        return
    writer.write(text)
    if text and not text.endswith("\n"):
        writer.write("\n")
    writer.flush()

def _format_tool_request(tool_request):
    name = tool_request.get("name") or "bash"
    if name == "bash":
        cmd = tool_request.get("command") or ""
        return f"bash: {cmd}".strip()
    args = []
    for key, value in tool_request.items():
        if key in ("id", "name", "arguments"): continue
        args.append(f"{key}={value}")
    args_text = " ".join(args).strip()
    return f"{name}: {args_text}".strip()

def _format_tool_result(tool_result):
    stdout = (tool_result.get("stdout") or "").rstrip("\n")
    stderr = (tool_result.get("stderr") or "").rstrip("\n")
    lines = []
    if stdout: lines.extend(stdout.splitlines())
    if stderr:
        if lines: lines.append("")
        lines.append("stderr:")
        lines.extend(stderr.splitlines())
    if not lines:
        message = (tool_result.get("message") or "").strip()
        lines.append(message if message else "OK")
    return "\n".join(lines)

def main():
    host, port, model, input_file, prompt = parse_args(sys.argv[1:])
    
    termprint("paragraph", "  ___   ___  __  __")
    termprint("paragraph", " / _ \\ | _ \\ \\ \\/ /")
    termprint("paragraph", "| (_) ||  _/  >  < ")
    termprint("paragraph", " \\___/ |_|   /_/\\_\\")
    termprint("paragraph", "")

    #termprint("paragraph", " _     _   _   ___            ___   ___  __  __")
    #termprint("paragraph", "| |_  | |_| |_ | _ \(_)  / / / _ \\ | _ \\ \\ \\/ /")
    #termprint("paragraph", "| ' \ |  _|  _||  _/    / / | (_) ||  _/  >  < ")
    #termprint("paragraph", "|_||_| \__|\__ |_|  (_)/ /  \\___/ |_|   /_/\\_\\")
    #termprint("paragraph", "")

    termprint("paragraph", f"Using model: {model} at {host}:{port}")
    termprint("paragraph", "")
    
    extra = ""
    if input_file: extra = input_file.read()
    if not sys.stdin.isatty(): extra = sys.stdin.read()
    if input_file: input_file.close()
    if extra: prompt = f"{prompt}\n\n```\n{extra}\n```"
    initial_prompt = prompt

    ensure_model_available(host, port, model)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    tools = [tool.describe() for tool in TOOL_INSTANCES]

    while True:
        body = json.dumps({"model": model, "messages": messages, "tools": tools, "stream": True})
        start_time = time.monotonic()
        
        termprint("paragraph", "\n")
        termprint("item", "Calling LLM to get tool instructions...")
        resp, _ = _http_request(host, port, "POST", "/v1/chat/completions", body=body, read_body=False, leave_open=True)
        if resp is None:
            termprint("error", f"Failed to connect to {host}:{port}")
            sys.exit(1)

        content_chunks = [] # we collect the content to use them in the fullfillment test
        stream_writer = _LinePrefixWriter(sys.stdout)
        tool_requests = process_response(resp, stream_writer, content_collector=content_chunks)
        resp.close()

        if tool_requests:
            tool_calls = []
            for tool_request in tool_requests:
                tool_name = tool_request.get("name")
                tool_calls.append(
                    {
                        "id": tool_request.get("id") or "call_1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": tool_request.get("arguments", "")},
                    }
                )
            messages.append({"role": "assistant", "tool_calls": tool_calls})
            for tool_request in tool_requests:
                tool_name = tool_request.get("name") or "bash"
                tool = TOOL_REGISTRY.get(tool_name, TOOL_REGISTRY["bash"])
                termprint("subitem", _format_tool_request(tool_request), title=f"Tool call: {tool_name}", elapsed=0.0)
                tool_start = time.monotonic()
                tool_result = tool.handle_request(tool_request)
                tool_elapsed = time.monotonic() - tool_start
                termprint("box", _format_tool_result(tool_result), title="Tool result", elapsed=tool_elapsed)
                tool_msg = {"role": "tool", "content": json.dumps(tool_result)}
                if tool_request.get("id"): tool_msg["tool_call_id"] = tool_request.get("id")
                messages.append(tool_msg)
            continue
        assistant_elapsed = time.monotonic() - start_time
        assistant_content = "".join(content_chunks).strip()
        if assistant_content and not assistant_content.endswith("\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()
        messages.append({"role": "assistant", "content": assistant_content})
        
        # test if task is fullfilled
        termprint("paragraph", "\n")
        termprint("item", "Testing the results so far if they fulfill the initial request...")
        conversation = _format_conversation(messages)
        judge_prompt = (
            "You are a strict judge. Return true only if the initial request is fully fulfilled "
            "by the conversation and the latest assistant response. Otherwise return false."
        )
        is_fulfilled, confidence = truth_test(
            judge_prompt,
            f"Conversation:\n{conversation}\n\nQuestion: was the initial request fulfilled?",
            compute_confidence=True, host=host, port=port, model=model)
        
        if (is_fulfilled):
            termprint("subitem", f"The initial request was fullfilled, confidence: {confidence * 100.0}%")
            break

        termprint("subitem", f"The initial request was not fullfilled; we compute a follow-up prompt to continue, confidence: {confidence * 100.0}%")
        follow_up_prompt = _build_follow_up_prompt(conversation, initial_prompt, host, port, model)
        
        if not follow_up_prompt:
            termprint("error", "no follow-up prompt computed; thats an error")
            break
        
        termprint("subitem", f"follow-up prompt: {follow_up_prompt}%")
        messages.append({"role": "user", "content": follow_up_prompt})
        continue

if __name__ == "__main__":
    main()
