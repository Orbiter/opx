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
    "You are a mighty Linux system operator. Make short answers. If code is requested, output only code."
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

def run_bash_tool(command):
    if not command: return {"tool": "bash", "exit_code": 1, "stdout": "", "stderr": "Empty command"}
    forbidden = ["|", ";", "&", ">", "<", "\n", "\r"]
    if any(token in command for token in forbidden):
        return {"tool": "bash", "exit_code": 1, "stdout": "", "stderr": "Rejected: unsafe command, try a safe approach"}
    sys.stderr.write(f"Tool request: {command}\nApprove? [Y/n]: ")
    sys.stderr.flush()
    resp = sys.stdin.readline()
    if resp and resp.strip().lower() not in ["", "y", "yes"]:
        return {"tool": "bash", "exit_code": 1, "stdout": "", "stderr": "Rejected by user, try a different approach"}
    try:
        args = shlex.split(command)
    except ValueError as exc:
        return {"tool": "bash", "exit_code": 1, "stdout": "", "stderr": str(exc)}
    if not args:
        return {"tool": "bash", "exit_code": 1, "stdout": "", "stderr": "Empty command"}
    try:
        completed = subprocess.run(args, capture_output=True, text=True)
    except OSError as exc:
        return {"tool": "bash", "exit_code": 1, "stdout": "", "stderr": str(exc)}
    return {
        "tool": "bash",
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

def run_edit_tool(diff_text):
    if not diff_text: return {"tool": "edit", "exit_code": 1, "stdout": "", "stderr": "Empty diff"}
    sys.stderr.write("Tool request (edit), diff:\n")
    sys.stderr.write(diff_text)
    if not diff_text.endswith("\n"): sys.stderr.write("\n")
    sys.stderr.write("Approve? [Y/n]: ")
    sys.stderr.flush()
    resp = sys.stdin.readline()
    if resp and resp.strip().lower() not in ["", "y", "yes"]:
        return {"tool": "edit", "exit_code": 1, "stdout": "", "stderr": "Rejected by user"}
    try:
        completed = subprocess.run(
            ["patch", "-p0", "--forward"],
            input=diff_text,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"tool": "edit", "exit_code": 1, "stdout": "", "stderr": str(exc)}
    return {
        "tool": "edit",
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

def run_write_tool(path, content):
    if not path: return {"tool": "write", "exit_code": 1, "stdout": "", "stderr": "Missing path"}
    if content is None: return {"tool": "write", "exit_code": 1, "stdout": "", "stderr": "Missing content"}
    sys.stderr.write("Tool request (write), path:\n")
    sys.stderr.write(f"{path}\n")
    sys.stderr.write("Content:\n")
    sys.stderr.write(content)
    if not content.endswith("\n"): sys.stderr.write("\n")
    sys.stderr.write("Approve? [Y/n]: ")
    sys.stderr.flush()
    resp = sys.stdin.readline()
    if resp and resp.strip().lower() not in ["", "y", "yes"]:
        return {"tool": "write", "exit_code": 1, "stdout": "", "stderr": "Rejected by user"}
    try:
        with open(path, "x", encoding="utf-8") as f:
            f.write(content)
    except FileExistsError:
        return {"tool": "write", "exit_code": 1, "stdout": "", "stderr": "File already exists"}
    except OSError as exc:
        return {"tool": "write", "exit_code": 1, "stdout": "", "stderr": str(exc)}
    return {"tool": "write", "exit_code": 0, "stdout": "OK\n", "stderr": ""}

def run_read_tool(path):
    if not path: return {"tool": "read", "exit_code": 1, "stdout": "", "stderr": "Missing path"}
    sys.stderr.write("Tool request (read), path:\n")
    sys.stderr.write(f"{path}\n")
    sys.stderr.write("Approve? [Y/n]: ")
    sys.stderr.flush()
    resp = sys.stdin.readline()
    if resp and resp.strip().lower() not in ["", "y", "yes"]:
        return {"tool": "read", "exit_code": 1, "stdout": "", "stderr": "Rejected by user"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
    except OSError as exc:
        return {"tool": "read", "exit_code": 1, "stdout": "", "stderr": str(exc)}
    return {"tool": "read", "exit_code": 0, "stdout": data, "stderr": ""}

def run_list_tool(path):
    if not path: return {"tool": "list", "exit_code": 1, "stdout": "", "stderr": "Missing path"}
    sys.stderr.write("Tool request (list), path:\n")
    sys.stderr.write(f"{path}\n")
    sys.stderr.write("Approve? [Y/n]: ")
    sys.stderr.flush()
    resp = sys.stdin.readline()
    if resp and resp.strip().lower() not in ["", "y", "yes"]:
        return {"tool": "list", "exit_code": 1, "stdout": "", "stderr": "Rejected by user"}
    try:
        entries = os.listdir(path)
    except OSError as exc:
        return {"tool": "list", "exit_code": 1, "stdout": "", "stderr": str(exc)}
    entries.sort()
    return {"tool": "list", "exit_code": 0, "stdout": "\n".join(entries) + "\n", "stderr": ""}

def run_mkdir_tool(path):
    if not path: return {"tool": "mkdir", "exit_code": 1, "stdout": "", "stderr": "Missing path"}
    sys.stderr.write("Tool request (mkdir), path:\n")
    sys.stderr.write(f"{path}\n")
    sys.stderr.write("Approve? [Y/n]: ")
    sys.stderr.flush()
    resp = sys.stdin.readline()
    if resp and resp.strip().lower() not in ["", "y", "yes"]:
        return {"tool": "mkdir", "exit_code": 1, "stdout": "", "stderr": "Rejected by user"}
    try:
        os.makedirs(path, exist_ok=False)
    except FileExistsError:
        return {"tool": "mkdir", "exit_code": 1, "stdout": "", "stderr": "Already exists"}
    except OSError as exc:
        return {"tool": "mkdir", "exit_code": 1, "stdout": "", "stderr": str(exc)}
    return {"tool": "mkdir", "exit_code": 0, "stdout": "OK\n", "stderr": ""}

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

def run_internet_read_tool(url):
    if not url: return {"tool": "internet_read", "exit_code": 1, "stdout": "", "stderr": "Missing url"}
    sys.stderr.write("Tool request (internet_read), url:\n")
    sys.stderr.write(f"{url}\n")
    sys.stderr.write("Approve? [Y/n]: ")
    sys.stderr.flush()
    resp = sys.stdin.readline()
    if resp and resp.strip().lower() not in ["", "y", "yes"]:
        return {"tool": "internet_read", "exit_code": 1, "stdout": "", "stderr": "Rejected by user"}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "opx/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if not _is_text_mime(content_type):
                return {"tool": "internet_read", "exit_code": 1, "stdout": "", "stderr": "Rejected: non-text mime type"}
            raw = resp.read()
    except urllib.error.URLError as exc:
        return {"tool": "internet_read", "exit_code": 1, "stdout": "", "stderr": str(exc)}
    charset = _parse_charset(content_type) or "utf-8"
    text = raw.decode(charset, errors="replace")
    if content_type.lower().startswith("text/html"):
        parser = HTMLToMarkdown()
        parser.feed(text)
        text = parser.get_markdown()
    return {"tool": "internet_read", "exit_code": 0, "stdout": text, "stderr": ""}

def _emit_content(content, writer, code_filter):
    if not content: return
    if code_filter:
        code_filter.feed(content)
        code_filter.flush()
    else:
        writer.write(content)
        writer.flush()

def process_response(resp, writer, code_only, stream):
    code_filter = CodeFilter(writer) if code_only else None
    tool_calls_data = {}
    tool_call_order = []

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
        if tool_name in ["bash", "edit", "write", "read", "list", "mkdir", "internet_read"] and tool_args:
            try:
                args_obj = json.loads(tool_args)
            except json.JSONDecodeError:
                return None
            if tool_name == "bash":
                command = args_obj.get("command")
                if isinstance(command, str):
                    return {"id": tool_id, "name": tool_name, "command": command, "arguments": tool_args}
            if tool_name == "edit":
                diff_text = args_obj.get("diff")
                if isinstance(diff_text, str):
                    return {"id": tool_id, "name": tool_name, "diff": diff_text, "arguments": tool_args}
            if tool_name == "write":
                path = args_obj.get("path")
                content = args_obj.get("content")
                if isinstance(path, str) and isinstance(content, str):
                    return {"id": tool_id, "name": tool_name, "path": path, "content": content, "arguments": tool_args}
            if tool_name in ["read", "list", "mkdir"]:
                path = args_obj.get("path")
                if isinstance(path, str):
                    return {"id": tool_id, "name": tool_name, "path": path, "arguments": tool_args}
            if tool_name == "internet_read":
                url = args_obj.get("url")
                if isinstance(url, str):
                    return {"id": tool_id, "name": tool_name, "url": url, "arguments": tool_args}
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
    tools = [
        {
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
        ,
        {
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
        ,
        {
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
        ,
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a text file and return its contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to a text file to read.",
                        }
                    },
                    "required": ["path"], "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list",
                "description": "List directory entries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list.",
                        }
                    },
                    "required": ["path"], "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mkdir",
                "description": "Create a new directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to create.",
                        }
                    },
                    "required": ["path"], "additionalProperties": False,
                },
                "strict": True,
            },
        }
        ,
        {
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
    ]

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
                if tool_request.get("name") == "edit":
                    tool_result = run_edit_tool(tool_request.get("diff", ""))
                    tool_name = "edit"
                elif tool_request.get("name") == "write":
                    tool_result = run_write_tool(
                        tool_request.get("path", ""),
                        tool_request.get("content", ""),
                    )
                    tool_name = "write"
                elif tool_request.get("name") == "read":
                    tool_result = run_read_tool(tool_request.get("path", ""))
                    tool_name = "read"
                elif tool_request.get("name") == "list":
                    tool_result = run_list_tool(tool_request.get("path", ""))
                    tool_name = "list"
                elif tool_request.get("name") == "mkdir":
                    tool_result = run_mkdir_tool(tool_request.get("path", ""))
                    tool_name = "mkdir"
                elif tool_request.get("name") == "internet_read":
                    tool_result = run_internet_read_tool(tool_request.get("url", ""))
                    tool_name = "internet_read"
                else:
                    tool_result = run_bash_tool(tool_request.get("command", ""))
                    tool_name = "bash"
                tool_msg = {"role": "tool", "content": json.dumps(tool_result)}
                if tool_request.get("id"):
                    tool_msg["tool_call_id"] = tool_request.get("id")
                messages.append(tool_msg)
            continue
        break

if __name__ == "__main__":
    main()
