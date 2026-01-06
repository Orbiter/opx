# opx
prompt-driven, system Operations Prompt eXecution

`opx` is a minimal command-line client for prompt-driven system operations using a local, OpenAI-compatible LLM endpoint.
We made this to make a simple, non-cloud, privacy-aware LLM-coding cli that can be applied to system operations.

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

The argument is always a prompt describing the intended operation.

---

## Minimal example

```
opx "add opx to the seek path"
```

---

## Notes

* `opx` does not execute commands automatically
* Shell commands requested by the LLM always require explicit user approval
* Network or execution errors are reported directly
