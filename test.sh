#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPX_PY="${OPX_PY:-$ROOT_DIR/opx.py}"
HOST="${OPX_HOST:-localhost}"
PORT="${OPX_PORT:-11434}"
MODEL="${OPX_MODEL:-qwen3:30b-a3b-instruct-2507-q4_K_M}"
#MODEL="${OPX_MODEL:-qwen3-vl:4b-instruct-q4_K_M}"
AUTO_APPROVE="${OPX_AUTO_APPROVE:-all}"
MAX_TURNS="${OPX_MAX_TURNS:-4}"
TOOL_TIMEOUT_SEC="${OPX_TOOL_TIMEOUT_SEC:-15}"
CALL_TIMEOUT_SEC="${CALL_TIMEOUT_SEC:-600}"
TARGET_PER_TOOL="${TARGET_PER_TOOL:-1}"
MAX_ATTEMPTS_PER_TOOL="${MAX_ATTEMPTS_PER_TOOL:-3}"
MAX_INFRA_RETRIES="${MAX_INFRA_RETRIES:-6}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-60}"
READY_POLL_SEC="${READY_POLL_SEC:-2}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/test-logs/$STAMP}"
TEST_ROOT="${TEST_ROOT:-/tmp/opx_tool_test}"

if [[ ! -f "$OPX_PY" ]]; then
  echo "error: opx.py not found at '$OPX_PY'" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

TOOLS=(
  tool_list bash git find grep edit_preview edit write read list tree man mkdir process_list network_scan internet_read
)

run_with_timeout() {
  local seconds="$1"
  shift
  python3 - "$seconds" "$@" <<'PY'
import subprocess
import sys

timeout_sec = int(sys.argv[1])
cmd = sys.argv[2:]
if not cmd:
    sys.exit(2)
try:
    completed = subprocess.run(cmd, timeout=timeout_sec)
    sys.exit(completed.returncode)
except subprocess.TimeoutExpired:
    print(f"timeout after {timeout_sec}s: {' '.join(cmd)}", file=sys.stderr)
    sys.exit(124)
PY
}

llm_ready() {
  python3 - "$HOST" "$PORT" <<'PY'
import http.client
import sys

host = sys.argv[1]
port = int(sys.argv[2])
try:
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/api/tags")
    resp = conn.getresponse()
    ok = 200 <= resp.status < 300
    resp.read()
    conn.close()
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
}

wait_for_llm_ready() {
  local timeout_sec="$1"
  local poll_sec="$2"
  local waited=0
  while (( waited < timeout_sec )); do
    if llm_ready; then
      return 0
    fi
    sleep "$poll_sec"
    waited=$((waited + poll_sec))
  done
  return 1
}

prepare_test_root() {
  rm -rf "$TEST_ROOT"
  mkdir -p "$TEST_ROOT"
  printf 'seed line 1\nseed line 2\n' > "$TEST_ROOT/test.txt"
  printf 'alpha\nbeta\ngamma\n' > "$TEST_ROOT/notes.txt"
}

prompt_for_tool() {
  local tool="$1"
  local round="$2"
  case "$tool" in
    tool_list)
      cat <<'P'
Call exactly one tool: tool_list. Then provide one short sentence and stop.
P
      ;;
    bash)
      cat <<'P'
Call exactly one tool: bash with command "pwd". Then provide one short sentence and stop.
P
      ;;
    git)
      cat <<P
Call exactly one tool: git with args ["status","-sb"] in repository path $ROOT_DIR (you can pass cwd if needed). Then provide one short sentence and stop.
P
      ;;
    find)
      cat <<P
Call exactly one tool and it must be "find".
Use args: path="$TEST_ROOT", name="*.txt".
Do not call bash or any other tool.
Then provide one short sentence and stop.
P
      ;;
    grep)
      cat <<P
Call exactly one tool: grep with pattern "seed" and path "$TEST_ROOT". Then provide one short sentence and stop.
P
      ;;
    edit_preview)
      cat <<P
Call exactly one tool: edit_preview with this unified diff and stop after one sentence:
--- $TEST_ROOT/test.txt
+++ $TEST_ROOT/test.txt
@@ -1,2 +1,2 @@
-seed line 1
+seed line 1 preview
 seed line 2
P
      ;;
    edit)
      cat <<P
Call exactly one tool: edit with this unified diff and stop after one sentence:
--- $TEST_ROOT/test.txt
+++ $TEST_ROOT/test.txt
@@ -1,2 +1,2 @@
 seed line 1
-seed line 2
+seed line 2 edited
P
      ;;
    write)
      cat <<P
Call exactly one tool and it must be "write".
Use args: path="$TEST_ROOT/generated_${round}.txt", content="generated round ${round}".
Do not call bash or any other tool.
Then provide one short sentence and stop.
P
      ;;
    read)
      cat <<P
Call exactly one tool: read with path "$TEST_ROOT/test.txt". Then provide one short sentence and stop.
P
      ;;
    list)
      cat <<P
Call exactly one tool: list with path "$TEST_ROOT". Then provide one short sentence and stop.
P
      ;;
    tree)
      cat <<P
Call exactly one tool: tree with path "$TEST_ROOT" and depth 1. Then provide one short sentence and stop.
P
      ;;
    man)
      cat <<'P'
Call exactly one tool: man with topic "ls". Then provide one short sentence and stop.
P
      ;;
    mkdir)
      cat <<P
Call exactly one tool: mkdir with path "$TEST_ROOT/newdir_${round}". Then provide one short sentence and stop.
P
      ;;
    process_list)
      cat <<'P'
Call exactly one tool: process_list with pattern "python". Then provide one short sentence and stop.
P
      ;;
    network_scan)
      cat <<'P'
Call exactly one tool: network_scan with target "localhost" and ports [22,80]. Then provide one short sentence and stop.
P
      ;;
    internet_read)
      cat <<'P'
Call exactly one tool: internet_read with url "https://example.com". Then provide one short sentence and stop.
P
      ;;
    *)
      echo "unknown tool: $tool" >&2
      return 1
      ;;
  esac
}

count_tool_calls() {
  local tool="$1"
  if compgen -G "$LOG_DIR/*.log" >/dev/null; then
    (rg -o "Tool call: ${tool}\\b" "$LOG_DIR"/*.log 2>/dev/null || true) | wc -l | tr -d '[:space:]'
  else
    echo 0
  fi
}

count_tool_passes() {
  local tool="$1"
  if compgen -G "$LOG_DIR/pass-${tool}-*.ok" >/dev/null; then
    ls "$LOG_DIR"/pass-"$tool"-*.ok 2>/dev/null | wc -l | tr -d '[:space:]'
  else
    echo 0
  fi
}

validate_attempt() {
  local tool="$1"
  local attempt="$2"
  local log_file="$3"
  local expected_write_file="$TEST_ROOT/generated_${attempt}.txt"

  # Every valid attempt must actually call the target tool.
  if ! rg -q "Tool call: ${tool}\\b" "$log_file"; then
    return 1
  fi

  case "$tool" in
    write)
      [[ -f "$expected_write_file" ]] || return 1
      [[ "$(cat "$expected_write_file")" == "generated round ${attempt}" ]] || return 1
      ;;
    edit)
      [[ -f "$TEST_ROOT/test.txt" ]] || return 1
      [[ "$(cat "$TEST_ROOT/test.txt")" == $'seed line 1\nseed line 2 edited' ]] || return 1
      ;;
    edit_preview)
      [[ -f "$TEST_ROOT/test.txt" ]] || return 1
      [[ "$(cat "$TEST_ROOT/test.txt")" == $'seed line 1\nseed line 2' ]] || return 1
      ;;
    find)
      rg -q "Tool request \\(find\\), path: ${TEST_ROOT}" "$log_file" || return 1
      if rg -q "stderr:" "$log_file"; then
        return 1
      fi
      ;;
    grep)
      rg -q "seed line" "$log_file" || return 1
      ;;
    read)
      rg -q "seed line 1" "$log_file" || return 1
      ;;
    list)
      rg -q "test\\.txt|notes\\.txt" "$log_file" || return 1
      ;;
    tree)
      rg -q "/tmp/opx_tool_test" "$log_file" || return 1
      ;;
    mkdir)
      [[ -d "$TEST_ROOT/newdir_${attempt}" ]] || return 1
      ;;
    process_list)
      rg -q "Tool call: process_list.*pattern=python|Tool request \\(process_list\\), pattern: python" "$log_file" || return 1
      rg -q "python|PID" "$log_file" || return 1
      ;;
    network_scan)
      rg -q "localhost|127\\.0\\.0\\.1|no open ports detected|/" "$log_file" || return 1
      ;;
    internet_read)
      rg -q "Example Domain|iana\\.org/domains/example" "$log_file" || return 1
      ;;
    bash)
      rg -q "/tmp/opx_tool_test|/Users" "$log_file" || return 1
      ;;
    git)
      rg -q "##|On branch|nothing to commit" "$log_file" || return 1
      ;;
    man)
      rg -q "LS\\(|NAME|man" "$log_file" || return 1
      ;;
    tool_list)
      rg -q "tool_list: List all available tools" "$log_file" || return 1
      ;;
  esac

  return 0
}

echo "Logs: $LOG_DIR"
echo "Running focused opx calls per tool until each tool reaches ${TARGET_PER_TOOL} calls."
echo "Test root: $TEST_ROOT"
echo "Approval mode: OPX_AUTO_APPROVE=$AUTO_APPROVE"
echo "Turn cap: OPX_MAX_TURNS=$MAX_TURNS"
echo "Tool timeout: OPX_TOOL_TIMEOUT_SEC=${TOOL_TIMEOUT_SEC}s"
echo "Call timeout: ${CALL_TIMEOUT_SEC}s"
echo "LLM readiness wait: ${READY_TIMEOUT_SEC}s (poll ${READY_POLL_SEC}s)"
echo

for tool in "${TOOLS[@]}"; do
  while :; do
    current=$(count_tool_passes "$tool")
    if (( current >= TARGET_PER_TOOL )); then
      break
    fi

    success=0
    infra_retries=0
    for attempt in $(seq 1 "$MAX_ATTEMPTS_PER_TOOL"); do
      if ! wait_for_llm_ready "$READY_TIMEOUT_SEC" "$READY_POLL_SEC"; then
        echo "ERROR LLM endpoint not ready at $HOST:$PORT after ${READY_TIMEOUT_SEC}s" >&2
        exit 2
      fi

      prepare_test_root
      prompt_text="$(prompt_for_tool "$tool" "$attempt")"
      log_file="$LOG_DIR/${tool}-$(date +%H%M%S)-a${attempt}.log"

      echo "=== Tool $tool (${current}/${TARGET_PER_TOOL}), attempt $attempt/$MAX_ATTEMPTS_PER_TOOL ==="
      set +e
      OPX_AUTO_APPROVE="$AUTO_APPROVE" OPX_MAX_TURNS="$MAX_TURNS" OPX_TOOL_TIMEOUT_SEC="$TOOL_TIMEOUT_SEC" OPX_ONLY_TOOLS="$tool" \
        run_with_timeout "$CALL_TIMEOUT_SEC" python3 "$OPX_PY" -h "$HOST" -p "$PORT" -m "$MODEL" "$prompt_text" 2>&1 | tee "$log_file"
      rc=${PIPESTATUS[0]}
      set -e
      echo "Saved: $log_file (rc=$rc)"

      if rg -q "Failed to connect to ${HOST}:${PORT}" "$log_file"; then
        infra_retries=$((infra_retries + 1))
        if (( infra_retries >= MAX_INFRA_RETRIES )); then
          echo "ERROR repeated connectivity failures to ${HOST}:${PORT} (${infra_retries} times)" >&2
          exit 2
        fi
        echo "warning: connectivity failure detected; retrying tool '$tool' without consuming progress" >&2
        sleep "$READY_POLL_SEC"
        continue
      fi

      if validate_attempt "$tool" "$attempt" "$log_file"; then
        : > "$LOG_DIR/pass-${tool}-$(date +%s)-$$-${RANDOM}-a${attempt}.ok"
        success=1
        break
      fi
    done

    if (( success == 0 )); then
      echo "warning: could not force tool '$tool' in ${MAX_ATTEMPTS_PER_TOOL} attempts" >&2
      break
    fi
  done
  echo
done

echo "=== Tool call totals ==="
missing=0
for tool in "${TOOLS[@]}"; do
  call_count=$(count_tool_calls "$tool")
  pass_count=$(count_tool_passes "$tool")
  printf '%-14s calls=%s validated=%s\n' "$tool" "$call_count" "$pass_count"
  if (( pass_count < TARGET_PER_TOOL )); then
    missing=1
  fi
done

if (( missing == 0 )); then
  echo "PASS: every tool reached ${TARGET_PER_TOOL} validated run(s)."
else
  echo "FAIL: some tools did not reach ${TARGET_PER_TOOL} validated run(s)." >&2
  echo "Inspect logs in: $LOG_DIR" >&2
  exit 1
fi
