#!/usr/bin/env bash
set -u
set -o pipefail

usage() {
  cat <<'USAGE'
Usage: opx.sh [options] <prompt>
Options:
  -m <model>      model name
  -h <host>       hostname
  -p <port>       port number
  -o <file>       write output to file instead of stdout
  -c              output ONLY code blocks
  -e <file>       read file content instead of stdin
  --help          show help and exit
USAGE
}

host="localhost"
port="11434"
model="devstral-small-2"
output_file=""
input_file=""
code_only=0

if [ $# -eq 0 ]; then
  usage >&2
  exit 1
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --help)
      usage
      exit 0
      ;;
    -m)
      shift
      [ $# -gt 0 ] || { usage >&2; exit 1; }
      model="$1"
      shift
      ;;
    -h)
      shift
      [ $# -gt 0 ] || { usage >&2; exit 1; }
      host="$1"
      shift
      ;;
    -p)
      shift
      [ $# -gt 0 ] || { usage >&2; exit 1; }
      port="$1"
      shift
      ;;
    -o)
      shift
      [ $# -gt 0 ] || { usage >&2; exit 1; }
      output_file="$1"
      shift
      ;;
    -c)
      code_only=1
      shift
      ;;
    -e)
      shift
      [ $# -gt 0 ] || { usage >&2; exit 1; }
      input_file="$1"
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      usage >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [ $# -eq 0 ]; then
  usage >&2
  exit 1
fi

prompt="$*"
extra=""

if [ -n "$input_file" ]; then
  if ! IFS= read -r -d '' extra <"$input_file"; then
    true
  fi
elif [ ! -t 0 ]; then
  if ! IFS= read -r -d '' extra; then
    true
  fi
fi

if [ -n "$extra" ]; then
  prompt="${prompt}"$'\n\n```\n'"${extra}"$'\n```'
fi

if [ -n "$input_file" ] && [ $code_only -eq 1 ]; then
  output_file="$input_file"
fi

stream="true"
if [ -n "$output_file" ]; then
  stream="false"
fi

output_fd=1
if [ -n "$output_file" ]; then
  exec 3>"$output_file"
  output_fd=3
fi

write_out() {
  printf '%s' "$1" >&$output_fd
}

out_escape=0
out_unicode_pending=0
out_unicode_hex=""

write_unescaped() {
  local chunk="$1"
  local i=0
  local len=${#chunk}
  local ch u
  while [ $i -lt $len ]; do
    ch="${chunk:$i:1}"
    if [ $out_unicode_pending -gt 0 ]; then
      case "$ch" in
        [0-9a-fA-F])
          out_unicode_hex+="$ch"
          out_unicode_pending=$((out_unicode_pending-1))
          if [ $out_unicode_pending -eq 0 ]; then
            printf -v u '\\u%s' "$out_unicode_hex"
            printf -v u '%b' "$u"
            write_out "$u"
            out_unicode_hex=""
          fi
          i=$((i+1))
          continue
          ;;
        *)
          write_out "\\u${out_unicode_hex}"
          out_unicode_hex=""
          out_unicode_pending=0
          ;;
      esac
    fi
    if [ $out_escape -eq 1 ]; then
      case "$ch" in
        'n') write_out $'\n' ;;
        'r') write_out $'\r' ;;
        't') write_out $'\t' ;;
        'u')
          out_unicode_pending=4
          out_unicode_hex=""
          out_escape=0
          i=$((i+1))
          continue
          ;;
        '\\') write_out "\\" ;;
        *) write_out "\\$ch" ;;
      esac
      out_escape=0
      i=$((i+1))
      continue
    fi
    if [ "$ch" = "\\" ]; then
      out_escape=1
      i=$((i+1))
      continue
    fi
    write_out "$ch"
    i=$((i+1))
  done
}

flush_unescaped() {
  if [ $out_unicode_pending -gt 0 ]; then
    write_out "\\u${out_unicode_hex}"
    out_unicode_hex=""
    out_unicode_pending=0
  fi
  if [ $out_escape -eq 1 ]; then
    write_out "\\"
    out_escape=0
  fi
}

output_chunk() {
  if [ $code_only -eq 1 ]; then
    filter_chunk "$1"
  else
    write_unescaped "$1"
  fi
}

flush_output() {
  if [ $code_only -eq 1 ]; then
    flush_filter
  else
    flush_unescaped
  fi
}

json_escape() {
  local s="$1"
  local out=""
  local i ch
  local len=${#s}
  for ((i=0; i<len; i++)); do
    ch="${s:$i:1}"
    case "$ch" in
      '\\') out+='\\\\' ;;
      '"') out+='\"' ;;
      $'\n') out+='\n' ;;
      $'\r') out+='\r' ;;
      $'\t') out+='\t' ;;
      $'\b') out+='\b' ;;
      $'\f') out+='\f' ;;
      *) out+="$ch" ;;
    esac
  done
  printf '%s' "$out"
}

repeat_backticks() {
  local n=$1
  local out=""
  local i=0
  while [ $i -lt $n ]; do
    out="${out}\`"
    i=$((i+1))
  done
  printf '%s' "$out"
}

in_code=0
bt_count=0
skip_lang=0
pending_sep=0

filter_chunk() {
  local chunk="$1"
  local i=0
  local len=${#chunk}
  local ch
  while [ $i -lt $len ]; do
    ch="${chunk:$i:1}"
    if [ $bt_count -gt 0 ]; then
      if [ "$ch" = '`' ]; then
        bt_count=$((bt_count+1))
        if [ $bt_count -eq 3 ]; then
          if [ $in_code -eq 0 ]; then
            in_code=1
            skip_lang=1
            if [ $pending_sep -eq 1 ]; then
              write_out $'\n'
              pending_sep=0
            fi
          else
            in_code=0
            pending_sep=1
          fi
          bt_count=0
        fi
        i=$((i+1))
        continue
      else
        if [ $in_code -eq 1 ] && [ $skip_lang -eq 0 ]; then
          write_out "$(repeat_backticks "$bt_count")"
        fi
        bt_count=0
        continue
      fi
    fi

    if [ "$ch" = '`' ]; then
      bt_count=1
      i=$((i+1))
      continue
    fi

    if [ $in_code -eq 1 ]; then
      if [ $skip_lang -eq 1 ]; then
        if [ "$ch" = $'\n' ]; then
          skip_lang=0
        fi
      else
        write_out "$ch"
      fi
    fi
    i=$((i+1))
  done
}

flush_filter() {
  if [ $in_code -eq 1 ] && [ $bt_count -gt 0 ] && [ $skip_lang -eq 0 ]; then
    write_out "$(repeat_backticks "$bt_count")"
  fi
  bt_count=0
}

json_extract_string_key() {
  local key="$1"
  local s="$2"
  case "$s" in
    *"\"$key\":"*) ;;
    *) return ;;
  esac
  local rest="${s#*\"$key\":}"
  while [ -n "$rest" ]; do
    case "${rest:0:1}" in
      $' ' | $'\t' | $'\n' | $'\r') rest="${rest:1}" ;;
      *) break ;;
    esac
  done
  if [ "${rest:0:1}" != '"' ]; then
    return
  fi
  rest="${rest:1}"
  local out=""
  local i=0
  local len=${#rest}
  local ch hex u
  local bs=0
  while [ $i -lt $len ]; do
    ch="${rest:$i:1}"
    if [ "$ch" = '\\' ]; then
      bs=$((bs+1))
      i=$((i+1))
      continue
    fi
    if [ "$ch" = '"' ] && [ $((bs % 2)) -eq 0 ]; then
      if [ $bs -gt 0 ]; then
        local pairs=$((bs / 2))
        local j=0
        while [ $j -lt $pairs ]; do
          out+="\\"
          j=$((j+1))
        done
      fi
      break
    fi
    if [ $bs -gt 0 ]; then
      local pairs=$((bs / 2))
      local odd=$((bs % 2))
      local j=0
      while [ $j -lt $pairs ]; do
        out+="\\"
        j=$((j+1))
      done
      if [ $odd -eq 1 ]; then
        case "$ch" in
          'n') out+=$'\n' ;;
          'r') out+=$'\r' ;;
          't') out+=$'\t' ;;
          'b') out+=$'\b' ;;
          'f') out+=$'\f' ;;
          '"') out+='"' ;;
          '\\') out+='\\' ;;
          '/') out+='/' ;;
          'u')
            if [ $((i+4)) -le $len ]; then
              hex="${rest:$((i+1)):4}"
              printf -v u '\\u%s' "$hex"
              printf -v u '%b' "$u"
              out+="$u"
              i=$((i+4))
            fi
            ;;
          *) out+="$ch" ;;
        esac
      else
        out+="$ch"
      fi
      bs=0
    else
      out+="$ch"
    fi
    i=$((i+1))
  done
  if [ $bs -gt 0 ]; then
    local j=0
    while [ $j -lt $bs ]; do
      out+="\\"
      j=$((j+1))
    done
  fi
  printf '%s' "$out"
}

json_extract_string_raw() {
  local key="$1"
  local s="$2"
  case "$s" in
    *"\"$key\":"*) ;;
    *) return ;;
  esac
  local rest="${s#*\"$key\":}"
  while [ -n "$rest" ]; do
    case "${rest:0:1}" in
      $' ' | $'\t' | $'\n' | $'\r') rest="${rest:1}" ;;
      *) break ;;
    esac
  done
  if [ "${rest:0:1}" != '"' ]; then
    return
  fi
  rest="${rest:1}"
  local out=""
  local esc=0
  local i=0
  local len=${#rest}
  local ch
  while [ $i -lt $len ]; do
    ch="${rest:$i:1}"
    if [ $esc -eq 1 ]; then
      out+="$ch"
      esc=0
    else
      if [ "$ch" = "\\" ]; then
        out+="\\"
        esc=1
      elif [ "$ch" = '"' ]; then
        break
      else
        out+="$ch"
      fi
    fi
    i=$((i+1))
  done
  printf '%s' "$out"
}

json_unescape() {
  local s="$1"
  local out=""
  local i=0
  local len=${#s}
  local ch u
  local esc=0
  local uni_pending=0
  local uni_hex=""
  while [ $i -lt $len ]; do
    ch="${s:$i:1}"
    if [ $uni_pending -gt 0 ]; then
      case "$ch" in
        [0-9a-fA-F])
          uni_hex+="$ch"
          uni_pending=$((uni_pending-1))
          if [ $uni_pending -eq 0 ]; then
            printf -v u '\\u%s' "$uni_hex"
            printf -v u '%b' "$u"
            out+="$u"
            uni_hex=""
          fi
          i=$((i+1))
          continue
          ;;
        *)
          out+="\\u${uni_hex}"
          uni_hex=""
          uni_pending=0
          ;;
      esac
    fi
    if [ $esc -eq 1 ]; then
      case "$ch" in
        'n') out+=$'\n' ;;
        'r') out+=$'\r' ;;
        't') out+=$'\t' ;;
        'u')
          uni_pending=4
          uni_hex=""
          esc=0
          i=$((i+1))
          continue
          ;;
        '\\') out+='\\' ;;
        '"') out+='"' ;;
        '/') out+='/' ;;
        'b') out+=$'\b' ;;
        'f') out+=$'\f' ;;
        *) out+="\\$ch" ;;
      esac
      esc=0
      i=$((i+1))
      continue
    fi
    if [ "$ch" = "\\" ]; then
      esc=1
      i=$((i+1))
      continue
    fi
    out+="$ch"
    i=$((i+1))
  done
  if [ $uni_pending -gt 0 ]; then
    out+="\\u${uni_hex}"
  fi
  if [ $esc -eq 1 ]; then
    out+="\\"
  fi
  printf '%s' "$out"
}

json_extract_content() {
  json_extract_string_key "content" "$1"
}

tool_name=""
tool_args=""
tool_args_unescaped=""
tool_id=""
tool_command=""

tool_reset() {
  tool_name=""
  tool_args=""
  tool_args_unescaped=""
  tool_id=""
  tool_command=""
}

tool_parse_args() {
  local args="$1"
  tool_args_unescaped="$(json_unescape "$args")"
  tool_command="$(json_extract_string_key "command" "$tool_args_unescaped")"
  [ -n "$tool_command" ]
}

is_safe_bash_command() {
  case "$1" in
    *'|'*|*';'*|*'&'*|*'>'*|*'<'*|*$'\n'*|*$'\r'*)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

bash_tool_approval() {
  local cmd="$1"
  printf 'Tool request: %s\nApprove? [Y/n]: ' "$cmd" >&2
  IFS= read -r reply
  case "${reply}" in
    ""|[Yy]|[Yy][Ee][Ss])
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

tool_exit_code=0
tool_stdout=""
tool_stderr=""

run_bash_tool() {
  local cmd="$1"
  tool_exit_code=1
  tool_stdout=""
  tool_stderr=""

  if [ -z "$cmd" ]; then
    tool_stderr="Empty command"
    return
  fi
  if ! is_safe_bash_command "$cmd"; then
    tool_stderr="Rejected: unsafe command, try a safe approach"
    return
  fi
  if ! bash_tool_approval "$cmd"; then
    tool_stderr="Rejected by user, try a different approach"
    return
  fi

  local stdout_capture=""
  stdout_capture="$(eval "$cmd" 2>&1)"
  tool_exit_code=$?
  tool_stdout="$stdout_capture"
  tool_stderr=""
}

append_message() {
  local role="$1"
  local content="$2"
  messages="${messages%]}"
  messages="${messages}, {\"role\":\"$role\",\"content\":\"$(json_escape "$content")\"}]"
}

system_prompt="Be a mighty Linux system operator. Use short answers. If code or commands are requested, output only code."
messages="[{\"role\":\"system\",\"content\":\"$(json_escape "$system_prompt")\"},{\"role\":\"user\",\"content\":\"$(json_escape "$prompt")\"}]"
tools="[{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"description\":\"Run a single shell command and return stdout/stderr.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\",\"description\":\"Single shell command (no pipes or redirection).\"}},\"required\":[\"command\"],\"additionalProperties\":false},\"strict\":true}}]"

url="http://$host:$port/v1/chat/completions"
while :; do
  tool_reset
  json="{\"model\":\"$(json_escape "$model")\",\"messages\":$messages,\"tools\":$tools,\"stream\":$stream}"

  if [ "$stream" = "true" ]; then
    status_code=""
    done_stream=0
    exec 3< <(curl -sS -N -H "Content-Type: application/json" -d "$json" -w $'\n__CURL_STATUS__%{http_code}\n' "$url")
    while IFS= read -r line <&3; do
      case "$line" in
        __CURL_STATUS__*)
          status_code="${line#__CURL_STATUS__}"
          break
          ;;
      esac
      if [ $done_stream -eq 1 ]; then
        continue
      fi
      case "$line" in
        data:*)
          data="${line#data:}"
          data="${data# }"
          if [ "$data" = "[DONE]" ]; then
            done_stream=1
            continue
          fi
          chunk="$(json_extract_content "$data")"
          if [ -n "$chunk" ]; then
            output_chunk "$chunk"
          fi
          if [ -z "$tool_name" ]; then
            tool_name="$(json_extract_string_key "name" "$data")"
          fi
          if [ -z "$tool_id" ]; then
            tool_id="$(json_extract_string_key "id" "$data")"
          fi
          args_part="$(json_extract_string_raw "arguments" "$data")"
          if [ -n "$args_part" ]; then
            tool_args="${tool_args}${args_part}"
            if [ -z "$tool_name" ] || [ "$tool_name" != "bash" ]; then
              tool_name="bash"
            fi
          fi
          ;;
        *)
          ;;
      esac
    done
    exec 3<&-
    if [ -z "$status_code" ] || [ "$status_code" -lt 200 ] || [ "$status_code" -ge 300 ]; then
      echo "Network error" >&2
      exit 1
    fi
  else
    response=""
    if ! response="$(curl -sS -f -H "Content-Type: application/json" -d "$json" "$url")"; then
      echo "Network error" >&2
      exit 1
    fi
    chunk="$(json_extract_content "$response")"
    if [ -n "$chunk" ]; then
      output_chunk "$chunk"
    fi
    tool_name="$(json_extract_string_key "name" "$response")"
    tool_id="$(json_extract_string_key "id" "$response")"
    tool_args="$(json_extract_string_raw "arguments" "$response")"
    if [ -n "$tool_args" ] && { [ -z "$tool_name" ] || [ "$tool_name" != "bash" ]; }; then
      tool_name="bash"
    fi
  fi

  if [ "$tool_name" = "bash" ] && [ -n "$tool_args" ]; then
    if tool_parse_args "$tool_args"; then
    run_bash_tool "$tool_command"
    tool_result_json="{\"tool\":\"bash\",\"exit_code\":$tool_exit_code,\"stdout\":\"$(json_escape "$tool_stdout")\",\"stderr\":\"$(json_escape "$tool_stderr")\"}"
    tool_call="{\"id\":\"${tool_id:-call_1}\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"$(json_escape "$tool_args_unescaped")\"}}"
    messages="${messages%]}"
    messages="${messages}, {\"role\":\"assistant\",\"tool_calls\":[$tool_call]}]"
    messages="${messages%]}"
    messages="${messages}, {\"role\":\"tool\",\"tool_call_id\":\"${tool_id:-call_1}\",\"content\":\"$(json_escape "$tool_result_json")\"}]"
    continue
    fi
  fi

  flush_output
  break
done
