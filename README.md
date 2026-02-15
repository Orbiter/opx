# opx
prompt-driven, system Operations Prompt eXecution

`opx` is a minimal command-line client for prompt-driven system operations using a local, OpenAI-compatible LLM endpoint.
We made this to make a simple, non-cloud, privacy-aware LLM-coding cli that can be applied to system operations.

`opx` is designed to work entirely on a local machine.  
It connects to a locally running LLM that exposes an OpenAI-compatible
`/v1/chat/completions` endpoint and optionally allows *human-approved*
execution of shell commands.

The program operates as follows:

1. The user provides a natural-language prompt on the command line
2. The prompt is sent to the local LLM together with a fixed system instruction
3. The LLM may:
   * return plain text
   * return code blocks
   * request execution of a **single shell command**
4. Any requested shell command is:
   * displayed to the user
   * executed **only after explicit approval**
   * restricted to a safe subset (no pipes, redirects, or chaining)
5. The command output is sent back to the LLM
6. The LLM may continue reasoning based on the result

---

## Requirements

* Linux or Unix-like system
* `bash` and `curl`
* Python 3 (for `opx.py`)
* A running **Ollama** instance
* A compatible model (**quen3:30b-a3b-instruct-2507-q4_K_M**, requires ~60GB VRAM)

By default, `opx` connects to:

* host: `localhost`
* port: `11434`

---

## Installation

1. Install and start **Ollama**

   Ensure that Ollama is running locally and listening on port `11434`.

2. Install the **qwen3:30b-a3b-instruct-2507-q4_K_M** model

   ```
   ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
   ```

3. Install `opx`

   Clone the repository and place either `opx.sh` or `opx.py` somewhere in your `PATH`:

   ```
   git clone https://github.com/Orbiter/opx
   cd opx
   chmod +x opx.py
   sudo cp opx.py /usr/local/bin/opx
   ```

   Alternatively, use `opx.sh` if you prefer shell.

---

## Usage

```
opx "<prompt>"
```

The argument is a natural-language prompt describing the intended operation.

### Command-line options

`opx` supports the following options:

```
opx [options] <prompt>
```

Options:

* `-m <model>`  
  Name of the LLM model to use.  
  Default: `llama3.2:3b`

* `-h <host>`  
  Hostname of the OpenAI-compatible API endpoint.  
  Default: `localhost`

* `-p <port>`  
  Port number of the API endpoint.  
  Default: `11434`

* `-e <file>`  
  Read file content instead of stdin.

* `--help`  
  Print a short usage summary and exit.

### Tool Execution

`opx` integrates with a rich suite of tools to perform system operations safely:

| Tool | Description |
|------|-------------|
| `bash` | Run a shell command via `/bin/bash` and return `stdout`/`stderr` |
| `git` | Run a safe, read-only `git` command and return `stdout`/`stderr` |
| `find` | Find files or directories starting at a path, optionally filtering by name, type, or depth |
| `grep` | Search files with `ripgrep` and return matching lines |
| `edit_preview` | Preview a unified diff without applying it |
| `edit` | Apply a unified diff to edit or patch files |
| `write` | Create or overwrite a file with provided content |
| `read` | Read a text file and return its contents |
| `list` | List directory entries |
| `tree` | Create a tree listing up to a maximum depth (1-3) |
| `man` | Read a system manual page |
| `mkdir` | Create a new directory |
| `process_list` | List running processes filtered by a required search pattern |
| `network_scan` | Scan a host or local network for IPs and common services |
| `internet_read` | Read a text resource from a URL; HTML is converted to Markdown |

All tool executions are **explicitly approved** by the user and are **not allowed to chain, redirect, or pipe**.

---

## Examples

```
opx "add opx to the seek path"
```

```
opx "show all running processes with 'python' in the name"
```

```
opx "create a new directory named 'project' and add a README.md file with 'Hello, world!' content"
```

```
opx "find all files named 'Dockerfile' in the current directory or subdirectories"
```

```
opx "read the README.md file"
```

```
opx "scan my local network for open ports 80, 443, and 3389"
```

---

## Notes

* `opx` does not execute commands automatically
* Shell commands requested by the LLM always require explicit user approval
* Network or execution errors are reported directly
* All tools are sandboxed and do not allow unsafe operations like file deletion or system reboots
* The model version `qwen3:30b-a3b-instruct-2507-q4_K_M` is recommended for best performance due to its large context window and tool-calling capability
* Environment variables like `OPX_AUTO_APPROVE` can be used to automate approvals (`read`, `write`, or `all`)
