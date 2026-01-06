#!/usr/bin/env python3
import sys
import json
import shlex
import subprocess
import http.client
import argparse

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
    "Be a mighty Linux system operator. Use short answers. If code or commands are requested, output only code."
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

def read_extra(input_fh):
    if input_fh: return input_fh.read()
    if not sys.stdin.isatty(): return sys.stdin.read()
    return ""

def build_prompt(prompt, extra):
    if extra: return f"{prompt}\n\n```\n{extra}\n```"
    return prompt

def send_request(host, port, body):
    conn = http.client.HTTPConnection(host, int(port), timeout=60)
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
    except (OSError, http.client.HTTPException):
        conn.close()
        return None, "Network error"
    return resp, None

def request_response(host, port, body):
    resp, err = send_request(host, port, body)
    if err:
        sys.stderr.write(err + "\n")
        sys.exit(1)
    if resp is None or resp.status < 200 or resp.status >= 300:
        sys.stderr.write("Network error\n")
        sys.exit(1)
    return resp

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
    tool_name = None
    tool_args = ""
    tool_id = None
    if stream:
        while True:
            line = resp.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            _emit_content(content, writer, code_filter)
            tool_calls = delta.get("tool_calls") or []
            for call in tool_calls:
                func = call.get("function") or {}
                name = func.get("name")
                arguments = func.get("arguments")
                if name and tool_name is None:
                    tool_name = name
                if call.get("id") and tool_id is None:
                    tool_id = call.get("id")
                if arguments:
                    tool_args += arguments
    else:
        body = resp.read()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            if code_filter:
                code_filter.flush()
            writer.flush()
            return None
        choices = payload.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            _emit_content(content, writer, code_filter)
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                call = tool_calls[0]
                tool_id = call.get("id")
                func = call.get("function") or {}
                tool_name = func.get("name")
                tool_args = func.get("arguments") or ""
    if code_filter: code_filter.flush()
    writer.flush()
    if tool_args and tool_name is None: tool_name = "bash"
    if tool_name == "bash" and tool_args:
        try:
            args_obj = json.loads(tool_args)
        except json.JSONDecodeError:
            return None
        command = args_obj.get("command")
        if isinstance(command, str):
            return {"id": tool_id, "name": tool_name, "command": command, "arguments": tool_args}
    return None

def main():
    host, port, model, output_file, input_file, code_only, prompt = parse_args(sys.argv[1:])
    input_name = ""
    if input_file: input_name = input_file.name
    extra = read_extra(input_file)
    if input_file: input_file.close()
    prompt = build_prompt(prompt, extra)

    output_path = ""
    if output_file:
        output_path = output_file.name
        output_file.close()

    if input_name and code_only and not output_path:
        output_path = input_name

    stream = output_path == ""
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
    ]

    while True:
        body = json.dumps({"model": model, "messages": messages, "tools": tools, "stream": stream})
        resp = request_response(host, port, body)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                tool_request = process_response(resp, f, code_only, stream)
        else:
            tool_request = process_response(resp, sys.stdout, code_only, stream)

        if tool_request:
            tool_result = run_bash_tool(tool_request.get("command", ""))
            tool_call = {
                "id": tool_request.get("id") or "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": tool_request.get("arguments", "")},
            }
            messages.append({"role": "assistant", "tool_calls": [tool_call]})
            tool_msg = {"role": "tool", "content": json.dumps(tool_result)}
            if tool_request.get("id"):
                tool_msg["tool_call_id"] = tool_request.get("id")
            messages.append(tool_msg)
            continue
        break

if __name__ == "__main__":
    main()
