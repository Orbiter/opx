#!/usr/bin/env python3
import os
import re
import sys
import json
import shlex
import argparse
import subprocess
import http.client
import urllib.error
import urllib.request
from html.parser import HTMLParser

USAGE = """Usage: opx.py [options] <prompt>
Options:
  -m <model>      model name
  -h <host>       hostname
  -p <port>       port number
  -o <file>       write output to file instead of stdout
  -c              output ONLY code blocks
  -e <file>       read file content instead of stdin
  --help          show help and exit
"""

SYSTEM_PROMPT = (
    "You are a mighty Linux system operator. Make short answers. If code is requested, output only code. Use tools as many times as you require to solve the given task."
)

def usage(exit_code=0, err=False):
    out = sys.stderr if err else sys.stdout
    out.write(USAGE)
    sys.exit(exit_code)

def parse_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-m", dest="model", default="llama3.2:3b")
    parser.add_argument("-h", dest="host", default="localhost")
    parser.add_argument("-p", dest="port", default="11434")
    parser.add_argument("-o", dest="output_file", type=argparse.FileType("w", encoding="utf-8"))
    parser.add_argument("-e", dest="input_file", type=argparse.FileType("r", encoding="utf-8"))
    parser.add_argument("-c", dest="code_only", action="store_true")
    parser.add_argument("--help", action="store_true")
    parser.add_argument("prompt", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.help: usage(0)
    prompt_parts = args.prompt
    if prompt_parts and prompt_parts[0] == "--": prompt_parts = prompt_parts[1:]
    if not prompt_parts: usage(1, True)

    prompt = " ".join(prompt_parts)
    return args.host, args.port, args.model, args.output_file, args.input_file, args.code_only, prompt

class CodeFilter:
    def __init__(self, writer):
        self.writer = writer
        self.in_code = False
        self.bt_count = 0
        self.skip_lang = False
        self.pending_sep = False

    def _write(self, text):
        if text: self.writer.write(text)

    def feed(self, text):
        for ch in text:
            while True:
                if self.bt_count:
                    if ch == "`":
                        self.bt_count += 1
                        if self.bt_count == 3:
                            if self.in_code:
                                self.in_code = False
                                self.pending_sep = True
                            else:
                                self.in_code = True
                                self.skip_lang = True
                                if self.pending_sep:
                                    self._write("\n")
                                    self.pending_sep = False
                            self.bt_count = 0
                        break
                    else:
                        if self.in_code and not self.skip_lang: self._write("`" * self.bt_count)
                        self.bt_count = 0
                        continue
                else:
                    if ch == "`":
                        self.bt_count = 1
                        break
                    if self.in_code:
                        if self.skip_lang:
                            if ch == "\n": self.skip_lang = False
                        else:
                            self._write(ch)
                    break

    def flush(self):
        if self.in_code and self.bt_count and not self.skip_lang: self._write("`" * self.bt_count)
        self.bt_count = 0

def request_response(host, port, body):
    conn = http.client.HTTPConnection(host, int(port), timeout=60)
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp is None or resp.status < 200 or resp.status >= 300:
            sys.stderr.write("Network error\n")
            sys.exit(1)
        return resp
    except (OSError, http.client.HTTPException):
        conn.close()
        sys.stderr.write("Network error\n")
        sys.exit(1)

def _ollama_request(host, port, method, path, body=None):
    conn = http.client.HTTPConnection(host, int(port), timeout=60)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        if resp is None or resp.status < 200 or resp.status >= 300:
            return None, None
        return resp, resp.read()
    except (OSError, http.client.HTTPException):
        return None, None
    finally:
        conn.close()

def ensure_model_available(host, port, model):
    resp, data = _ollama_request(host, port, "GET", "/api/tags")
    if not resp or not data: return
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return
    models = payload.get("models") or []
    if any(m.get("name") == model for m in models):
        return
    pull_body = json.dumps({"name": model, "stream": False})
    resp, _ = _ollama_request(host, port, "POST", "/api/pull", pull_body)
    if not resp:
        sys.stderr.write(f"Failed to pull model: {model}\n")
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
    sys.stderr.write(f"Approve '{subject_line}', {default_prompt} (a=always):")
    sys.stderr.flush()
    resp = sys.stdin.readline().strip().lower()
    if resp == "a":
        if write_request:
            approval_always_write = True
        else:
            approval_always_read = True
        return True
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
        rejectedmsg = "Rejected by user, try a different approach"
        if any(token in command for token in forbidden):
            return {"tool": "bash", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Rejected: unsafe command, try a safe approach", "data": {}, "message": "Rejected: unsafe command, try a safe approach"}
        if not request_tool_approval(f"bash: {command}", write_request=True):
            return {"tool": "bash", "ok": False, "exit_code": 1, "stdout": "", "stderr": rejectedmsg, "data": {}, "message": rejectedmsg}
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
        sys.stderr.write("Tool request (edit), diff:\n")
        sys.stderr.write(diff)
        if not diff.endswith("\n"):
            sys.stderr.write("\n")
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
        sys.stderr.write("Tool request (write), path:\n")
        sys.stderr.write(f"{path}\n")
        sys.stderr.write("Content:\n")
        sys.stderr.write(content)
        if not content.endswith("\n"):
            sys.stderr.write("\n")
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
        if not path:
            return {"tool": "read", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Missing path", "data": {}, "message": "Missing path"}
        sys.stderr.write("Tool request (read), path:\n")
        sys.stderr.write(f"{path}\n")
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
        if not path:
            return {"tool": "list", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Missing path", "data": {}, "message": "Missing path"}
        sys.stderr.write("Tool request (list), path:\n")
        sys.stderr.write(f"{path}\n")
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
        if not path:
            return {"tool": "mkdir", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Missing path", "data": {}, "message": "Missing path"}
        sys.stderr.write("Tool request (mkdir), path:\n")
        sys.stderr.write(f"{path}\n")
        if not request_tool_approval(f"mkdir: {path}", write_request=True):
            return {"tool": "mkdir", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Rejected by user", "data": {}, "message": "Rejected by user"}
        try:
            os.makedirs(path, exist_ok=False)
        except FileExistsError:
            return {"tool": "mkdir", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Already exists", "data": {}, "message": "Already exists"}
        except OSError as exc:
            return {"tool": "mkdir", "ok": False, "exit_code": 1, "stdout": "", "stderr": str(exc), "data": {}, "message": str(exc)}
        return {"tool": "mkdir", "ok": True, "exit_code": 0, "stdout": "OK\n", "stderr": "", "data": {"path": path}}

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
            if not self.in_pre:
                self.out.append("`")
        elif tag == "pre":
            self._ensure_blankline()
            self.out.append("```\n")
            self.in_pre = True
        elif tag == "a":
            self.link_href = None
            self.link_text = []
            for key, value in attrs:
                if key == "href":
                    self.link_href = value

    def handle_endtag(self, tag):
        if tag in ["strong", "b"]:
            self.out.append("**")
        elif tag in ["em", "i"]:
            self.out.append("*")
        elif tag == "code":
            if not self.in_pre:
                self.out.append("`")
        elif tag == "pre":
            if not "".join(self.out).endswith("\n"):
                self.out.append("\n")
            self.out.append("```\n")
            self.in_pre = False
        elif tag == "a":
            text = "".join(self.link_text).strip()
            href = self.link_href or ""
            if text:
                self.out.append(f"[{text}]({href})")
            self.link_href = None
            self.link_text = []

    def handle_data(self, data):
        if not data:
            return
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
        if not url:
            return {"tool": "internet_read", "ok": False, "exit_code": 1, "stdout": "", "stderr": "Missing url", "data": {}, "message": "Missing url"}
        sys.stderr.write("Tool request (internet_read), url:\n")
        sys.stderr.write(f"{url}\n")
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

def _emit_content(content, writer, code_filter):
    if not content: return
    if code_filter:
        code_filter.feed(content)
        code_filter.flush()
    else:
        writer.write(content)
        writer.flush()

def process_response(resp, writer, code_only, stream, tool_registry=None):
    code_filter = CodeFilter(writer) if code_only else None
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

    if stream:
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
            _emit_content(content, writer, code_filter)
            tool_calls = delta.get("tool_calls") or []
            for call in tool_calls:
                _add_or_update_tool_call(call)
    else:
        body = resp.read()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            if code_filter: code_filter.flush()
            writer.flush()
            return None
        choices = payload.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            _emit_content(content, writer, code_filter)
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    _add_or_update_tool_call(call)
    if code_filter: code_filter.flush()
    writer.flush()
    tool_requests = []
    for key in tool_call_order:
        tool_request = _build_tool_request(tool_calls_data.get(key, {}))
        if tool_request:
            tool_requests.append(tool_request)
    return tool_requests

def main():
    host, port, model, output_file, input_file, code_only, prompt = parse_args(sys.argv[1:])
    input_name = ""
    if input_file: input_name = input_file.name
    extra = ""
    if input_file: extra = input_file.read()
    if not sys.stdin.isatty(): extra = sys.stdin.read()
    if input_file: input_file.close()
    if extra: prompt = f"{prompt}\n\n```\n{extra}\n```"

    output_path = ""
    if output_file:
        output_path = output_file.name
        output_file.close()

    if input_name and code_only and not output_path:
        output_path = input_name

    stream = output_path == ""
    ensure_model_available(host, port, model)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    tools = [tool.describe() for tool in TOOL_INSTANCES]

    while True:
        body = json.dumps({"model": model, "messages": messages, "tools": tools, "stream": stream})
        resp = request_response(host, port, body)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                tool_requests = process_response(resp, f, code_only, stream)
        else:
            tool_requests = process_response(resp, sys.stdout, code_only, stream)

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
                tool_result = tool.handle_request(tool_request)
                tool_msg = {"role": "tool", "content": json.dumps(tool_result)}
                if tool_request.get("id"):
                    tool_msg["tool_call_id"] = tool_request.get("id")
                messages.append(tool_msg)
            continue
        break

if __name__ == "__main__":
    main()
