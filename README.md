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
* A compatible model (**TBD**, must support chat completions and tool calling)

By default, `opx` connects to:

* host: `localhost`
* port: `11434`

---

## Installation

1. Install and start **Ollama**

   Ensure that Ollama is running locally and listening on port `11434`.

2. Install a compatible model

   The required model is **TBD**. Once decided, it must be pulled via Ollama.

3. Install `opx`

   Clone the repository and place either `opx.sh` or `opx.py` somewhere in your `PATH`:

   ```
   git clone https://github.com/Orbiter/opx
   cd opx
   chmod +x opx.sh
   sudo cp opx.sh /usr/local/bin/opx
   ```

   Alternatively, use `opx.py` if you prefer Python.

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

* `--help`  
  Print a short usage summary and exit.


Examples of valid prompts:

* describing a system task
* asking for shell commands
* asking for code snippets
* asking the LLM to inspect files or system state (via approved commands)


### Tool execution

If the LLM requests execution of a shell command:

* the command is printed to stderr
* the user is asked for confirmation
* the command is executed locally using `subprocess`
* stdout, stderr, and exit code are returned to the LLM

Only a single command without pipes or redirection is allowed.

---

## Examples

```
opx "add opx to the seek path"
```

---

## Notes

* `opx` does not execute commands automatically
* Shell commands requested by the LLM always require explicit user approval
* Network or execution errors are reported directly
