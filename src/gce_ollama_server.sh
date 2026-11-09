#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Provisions and manages a dedicated GPU-accelerated GCE instance running Ollama
# to host the Gemma 4 26B MoE model (gemma4:26b) for end-to-end Sensemaking runs.
#
# Usage:
#   bash src/gce_ollama_server.sh up
#   bash src/gce_ollama_server.sh run-e2e <QUALTRICS_CSV> <WORKING_DIR> [MODEL_NAME]
#   bash src/gce_ollama_server.sh status
#   bash src/gce_ollama_server.sh tunnel
#   bash src/gce_ollama_server.sh stop
#   bash src/gce_ollama_server.sh delete

set -euo pipefail

PROJECT="${GCE_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
INSTANCE_NAME="${GCE_INSTANCE_NAME:-sensemaking-ollama-26b}"
REGION="${GCE_REGION:-us-central1}"
CANDIDATE_ZONES="${GCE_ZONES:-${REGION}-a,${REGION}-b,${REGION}-c,${REGION}-f}"
ZONE="${GCE_ZONE:-}"

# Model & Ollama tuning parameters
MODEL_NAME="${MODEL_NAME:-gemma4:26b}"
LOCAL_PORT="${LOCAL_PORT:-11435}"
REMOTE_PORT="11434"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-65536}"
OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-8192}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1m}"

# Pre-built Deep Learning VM image with NVIDIA Driver 580 + CUDA 12.9
IMAGE_PROJECT="${GCE_IMAGE_PROJECT:-deeplearning-platform-release}"
IMAGE_FAMILY="${GCE_IMAGE_FAMILY:-common-cu129-ubuntu-2404-nvidia-580}"
BOOT_DISK_SIZE="${GCE_BOOT_DISK_SIZE:-200GB}"
BOOT_DISK_TYPE="${GCE_BOOT_DISK_TYPE:-pd-ssd}"

# Lifecycle control after `run-e2e`:
#   STOP_ON_EXIT=1   -> Stops the GCE VM after run-e2e finishes (default: 0)
#   DELETE_ON_EXIT=1 -> Deletes the GCE VM after run-e2e finishes (default: 0)
STOP_ON_EXIT="${STOP_ON_EXIT:-0}"
DELETE_ON_EXIT="${DELETE_ON_EXIT:-0}"

TUNNEL_PID_FILE="/tmp/gce_ollama_tunnel_${INSTANCE_NAME}_${LOCAL_PORT}.pid"
ACTIVE_HOST_FILE="/tmp/gce_ollama_active_host"

usage() {
  cat <<EOF
Usage: $0 <command> [args...]

Commands:
  up                                         Create/start GCE GPU VM, install Ollama, pull ${MODEL_NAME}, and open SSH tunnel
  run-e2e <CSV> <WORKING_DIR> [MODEL_NAME]   Ensure GCE Ollama server is up and run src/test_sensemaking_e2e.sh against it
  tunnel                                     Start or verify local SSH port-forward (127.0.0.1:${LOCAL_PORT} -> VM:${REMOTE_PORT})
  status                                     Show GCE VM status, GPU utilization (nvidia-smi), loaded models, and tunnel status
  monitor                                    Live-watch GPU utilization, memory bandwidth, active requests, and recent Ollama request latencies
  logs                                       Stream live Ollama server logs (journalctl -u ollama -f) from the GCE VM
  stop                                       Close SSH tunnel and stop GCE VM (preserves disk & downloaded model weights)
  delete                                     Close SSH tunnel and permanently delete GCE VM

Environment Overrides:
  GCE_PROJECT          GCP project ID (default: ${PROJECT})
  GCE_INSTANCE_NAME    VM instance name (default: ${INSTANCE_NAME})
  GCE_ZONE             Explicit GCE zone (otherwise searches: ${CANDIDATE_ZONES})
  GCE_MACHINE_TYPE     Explicit machine type (otherwise tries a2-highgpu-1g -> g2-standard-24 -> g2-standard-12)
  MODEL_NAME           Ollama model tag to pull and serve (default: ${MODEL_NAME})
  LOCAL_PORT           Local port for SSH forwarding (default: ${LOCAL_PORT})
  OLLAMA_NUM_PARALLEL  Parallel request slots on GPU (default: ${OLLAMA_NUM_PARALLEL})
  OLLAMA_CONTEXT_LENGTH Context window length (default: ${OLLAMA_CONTEXT_LENGTH})
  OPENAI_MAX_TOKENS    Max completion tokens per request (default: ${OPENAI_MAX_TOKENS})
  STOP_ON_EXIT         Set to 1 to stop the VM automatically after 'run-e2e' completes
  DELETE_ON_EXIT       Set to 1 to delete the VM automatically after 'run-e2e' completes
EOF
}

log() {
  printf "\n[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

find_existing_instance_zone() {
  if [[ -n "$ZONE" ]]; then
    if gcloud compute instances describe "$INSTANCE_NAME" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
      echo "$ZONE"
      return 0
    fi
  fi
  gcloud compute instances list \
    --project="$PROJECT" \
    --filter="name=(${INSTANCE_NAME})" \
    --format="value(zone.basename())" 2>/dev/null | head -n 1
}

get_instance_status() {
  local zone="$1"
  gcloud compute instances describe "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$zone" \
    --format="value(status)" 2>/dev/null || true
}

create_or_start_instance() {
  local existing_zone
  existing_zone="$(find_existing_instance_zone)"

  if [[ -n "$existing_zone" ]]; then
    ZONE="$existing_zone"
    local status
    status="$(get_instance_status "$ZONE")"
    log "Found existing instance '${INSTANCE_NAME}' in zone '${ZONE}' (status: ${status})."
    if [[ "$status" != "RUNNING" ]]; then
      log "Starting instance '${INSTANCE_NAME}' in zone '${ZONE}'..."
      gcloud compute instances start "$INSTANCE_NAME" --project="$PROJECT" --zone="$ZONE"
    fi
    return 0
  fi

  # Accelerator profiles ordered by performance for 26B MoE (~18 GB weights + parallel KV cache):
  # 1. a2-highgpu-1g  : 1x NVIDIA A100 40GB HBM2 (1,555 GB/s bandwidth, single-GPU zero split overhead)
  # 2. g2-standard-24 : 2x NVIDIA L4 48GB total GDDR6 (48 GB VRAM for large parallel KV cache)
  # 3. g2-standard-12 : 1x NVIDIA L4 24GB GDDR6 (widest zone availability)
  local -a profiles
  if [[ -n "${GCE_MACHINE_TYPE:-}" ]]; then
    profiles=("${GCE_MACHINE_TYPE}")
  else
    profiles=("a2-highgpu-1g" "g2-standard-24" "g2-standard-12")
  fi

  local -a zones
  if [[ -n "$ZONE" ]]; then
    zones=("$ZONE")
  else
    IFS=',' read -r -a zones <<< "$CANDIDATE_ZONES"
  fi

  for machine_type in "${profiles[@]}"; do
    for candidate_zone in "${zones[@]}"; do
      log "Attempting to create '${INSTANCE_NAME}' (${machine_type}) in '${candidate_zone}'..."
      if gcloud compute instances create "$INSTANCE_NAME" \
        --project="$PROJECT" \
        --zone="$candidate_zone" \
        --machine-type="$machine_type" \
        --maintenance-policy=TERMINATE \
        --restart-on-failure \
        --image-project="$IMAGE_PROJECT" \
        --image-family="$IMAGE_FAMILY" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --boot-disk-type="$BOOT_DISK_TYPE" \
        --scopes=default; then
        ZONE="$candidate_zone"
        log "Successfully created '${INSTANCE_NAME}' (${machine_type}) in '${ZONE}'."
        return 0
      else
        log "Could not allocate ${machine_type} in ${candidate_zone}; trying next option..."
      fi
    done
  done

  echo "Error: Failed to provision GCE GPU instance '${INSTANCE_NAME}' across all candidate profiles and zones." >&2
  exit 1
}

wait_for_ssh() {
  log "Waiting for SSH and NVIDIA GPU driver readiness on '${INSTANCE_NAME}' (${ZONE})..."
  for attempt in {1..30}; do
    if gcloud compute ssh "$INSTANCE_NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --command="nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null; then
      return 0
    fi
    sleep 5
  done
  echo "Error: Timed out waiting for SSH / GPU readiness on '${INSTANCE_NAME}'." >&2
  exit 1
}

provision_remote_ollama() {
  log "Configuring Ollama server and pulling '${MODEL_NAME}' on '${INSTANCE_NAME}' (${ZONE})..."
  gcloud compute ssh "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --command="bash -s -- '${MODEL_NAME}' '${REMOTE_PORT}' '${OLLAMA_NUM_PARALLEL}' '${OLLAMA_CONTEXT_LENGTH}' '${OLLAMA_KEEP_ALIVE}'" <<'REMOTE_EOF'
set -euo pipefail

MODEL_NAME="$1"
REMOTE_PORT="$2"
OLLAMA_NUM_PARALLEL="$3"
OLLAMA_CONTEXT_LENGTH="$4"
OLLAMA_KEEP_ALIVE="$5"

echo "=== GPU Hardware Detected ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

if ! command -v ollama >/dev/null 2>&1; then
  echo "=== Installing Ollama ==="
  if ! command -v zstd >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y zstd
  fi
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "=== Configuring systemd service for GPU-accelerated Ollama ==="
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<EOF
[Service]
Environment="OLLAMA_HOST=127.0.0.1:${REMOTE_PORT}"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}"
Environment="OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH}"
Environment="OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_MAX_QUEUE=512"
EOF

sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl restart ollama

echo "=== Waiting for remote Ollama daemon on 127.0.0.1:${REMOTE_PORT} ==="
export OLLAMA_HOST="127.0.0.1:${REMOTE_PORT}"
for _ in {1..30}; do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! ollama list | awk -v model="$MODEL_NAME" 'NR>1 && ($1 == model || $1 == model ":latest") {found=1} END {exit !found}'; then
  echo "=== Pulling ${MODEL_NAME} ==="
  ollama pull "$MODEL_NAME"
else
  echo "=== Model ${MODEL_NAME} is already present on disk ==="
fi

echo "=== Warming up ${MODEL_NAME} into GPU VRAM ==="
curl -sf "http://127.0.0.1:${REMOTE_PORT}/api/generate" \
  -d "{\"model\": \"${MODEL_NAME}\", \"prompt\": \"Hello\", \"stream\": false, \"keep_alive\": \"${OLLAMA_KEEP_ALIVE}\", \"options\": {\"num_ctx\": ${OLLAMA_CONTEXT_LENGTH}, \"num_predict\": 1}}" >/dev/null

echo "=== Active Ollama Models (ollama ps) ==="
ollama ps
REMOTE_EOF
}

is_tunnel_active() {
  if [[ -f "$TUNNEL_PID_FILE" ]]; then
    local pid
    pid="$(cat "$TUNNEL_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      if curl -sf "http://127.0.0.1:${LOCAL_PORT}/api/tags" >/dev/null 2>&1; then
        return 0
      fi
    fi
  fi
  return 1
}

stop_tunnel() {
  if [[ -f "$TUNNEL_PID_FILE" ]]; then
    local pid
    pid="$(cat "$TUNNEL_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      log "Stopping SSH tunnel (PID ${pid})..."
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$TUNNEL_PID_FILE"
  fi
  rm -f "$ACTIVE_HOST_FILE"
}

start_tunnel() {
  if [[ -z "$ZONE" ]]; then
    ZONE="$(find_existing_instance_zone)"
  fi
  if [[ -z "$ZONE" ]]; then
    echo "Error: Instance '${INSTANCE_NAME}' not found in project '${PROJECT}'. Run '$0 up' first." >&2
    exit 1
  fi

  if is_tunnel_active; then
    echo "127.0.0.1:${LOCAL_PORT}" > "$ACTIVE_HOST_FILE"
    log "SSH tunnel to '${INSTANCE_NAME}' is already active at http://127.0.0.1:${LOCAL_PORT}."
    return 0
  fi

  stop_tunnel

  log "Opening background SSH tunnel 127.0.0.1:${LOCAL_PORT} -> ${INSTANCE_NAME}:${REMOTE_PORT} (${ZONE})..."
  gcloud compute ssh "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    -- -N -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 >/dev/null 2>&1 &
  local tunnel_pid=$!
  echo "$tunnel_pid" > "$TUNNEL_PID_FILE"

  for _ in {1..30}; do
    if is_tunnel_active; then
      echo "127.0.0.1:${LOCAL_PORT}" > "$ACTIVE_HOST_FILE"
      log "SSH tunnel established (PID ${tunnel_pid}) at http://127.0.0.1:${LOCAL_PORT}."
      return 0
    fi
    if ! kill -0 "$tunnel_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  echo "Error: Failed to establish SSH port-forward tunnel on 127.0.0.1:${LOCAL_PORT} (port may already be in use)." >&2
  stop_tunnel
  exit 1
}

cmd_up() {
  create_or_start_instance
  wait_for_ssh
  provision_remote_ollama
  start_tunnel
  log "Dedicated GCE Ollama server is ready!"
  echo "  OLLAMA_HOST=127.0.0.1:${LOCAL_PORT}"
  echo "  MODEL_NAME=${MODEL_NAME}"
  echo ""
  echo "To run the Sensemaking E2E test against this server:"
  echo "  bash $0 run-e2e src/smoke_test_data.csv /tmp/sensemaking_e2e_output ${MODEL_NAME}"
}

cmd_status() {
  local zone
  zone="$(find_existing_instance_zone)"
  if [[ -z "$zone" ]]; then
    echo "Instance '${INSTANCE_NAME}' does not exist in project '${PROJECT}'."
    return 0
  fi
  local status
  status="$(get_instance_status "$zone")"
  echo "Instance: ${INSTANCE_NAME} (zone: ${zone}, status: ${status})"
  if is_tunnel_active; then
    echo "Tunnel:   ACTIVE on 127.0.0.1:${LOCAL_PORT} (PID $(cat "$TUNNEL_PID_FILE"))"
  else
    echo "Tunnel:   INACTIVE on 127.0.0.1:${LOCAL_PORT}"
  fi
  if [[ "$status" == "RUNNING" ]]; then
    echo ""
    gcloud compute ssh "$INSTANCE_NAME" \
      --project="$PROJECT" \
      --zone="$zone" \
      --command="nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,power.draw,memory.used,memory.total --format=csv && echo '' && OLLAMA_HOST=127.0.0.1:${REMOTE_PORT} ollama ps"
  fi
}

cmd_monitor() {
  local zone
  zone="$(find_existing_instance_zone)"
  if [[ -z "$zone" ]]; then
    echo "Error: Instance '${INSTANCE_NAME}' does not exist in project '${PROJECT}'." >&2
    exit 1
  fi
  gcloud compute ssh "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$zone" \
    -- -t "watch -n 2 'echo \"=== GPU Utilization & Memory Bandwidth ===\"; nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,power.draw,temperature.gpu,memory.used,memory.total --format=csv; echo \"\"; echo \"=== Loaded Ollama Models ===\"; OLLAMA_HOST=127.0.0.1:${REMOTE_PORT} ollama ps; echo \"\"; echo \"=== Active In-Flight HTTP Connections ===\"; ss -tn state established \"( sport = :${REMOTE_PORT} )\" | tail -n +2 | wc -l; echo \"\"; echo \"=== Recent Ollama Requests (GIN Latency Log) ===\"; journalctl -u ollama -n 12 --no-pager | tail -n 12'"
}

cmd_logs() {
  local zone
  zone="$(find_existing_instance_zone)"
  if [[ -z "$zone" ]]; then
    echo "Error: Instance '${INSTANCE_NAME}' does not exist in project '${PROJECT}'." >&2
    exit 1
  fi
  gcloud compute ssh "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$zone" \
    --command="journalctl -u ollama -f"
}

cmd_stop() {
  stop_tunnel
  local zone
  zone="$(find_existing_instance_zone)"
  if [[ -z "$zone" ]]; then
    log "Instance '${INSTANCE_NAME}' does not exist."
    return 0
  fi
  local status
  status="$(get_instance_status "$zone")"
  if [[ "$status" == "RUNNING" ]]; then
    log "Stopping GCE instance '${INSTANCE_NAME}' in '${zone}'..."
    gcloud compute instances stop "$INSTANCE_NAME" --project="$PROJECT" --zone="$zone"
  else
    log "Instance '${INSTANCE_NAME}' is already in state '${status}'."
  fi
}

cmd_delete() {
  stop_tunnel
  local zone
  zone="$(find_existing_instance_zone)"
  if [[ -z "$zone" ]]; then
    log "Instance '${INSTANCE_NAME}' does not exist."
    return 0
  fi
  log "Deleting GCE instance '${INSTANCE_NAME}' in '${zone}'..."
  gcloud compute instances delete "$INSTANCE_NAME" --project="$PROJECT" --zone="$zone" --quiet
}

cmd_run_e2e() {
  if [[ "$#" -lt 2 ]]; then
    echo "Usage: $0 run-e2e <QUALTRICS_CSV> <WORKING_DIR> [MODEL_NAME] [FIXED_RESPONSE_COLS] [FOLLOW_UP_QUESTION_COLS] [FOLLOW_UP_RESPONSE_COLS]" >&2
    exit 1
  fi

  local qualtrics_csv="$1"
  local working_dir="$2"
  local model_name="${3:-$MODEL_NAME}"
  shift 2
  if [[ "$#" -ge 1 ]]; then
    shift 1
  fi

  MODEL_NAME="$model_name"
  cmd_up

  cleanup_after_e2e() {
    if [[ "$DELETE_ON_EXIT" == "1" ]]; then
      cmd_delete
    elif [[ "$STOP_ON_EXIT" == "1" ]]; then
      cmd_stop
    fi
  }
  trap cleanup_after_e2e EXIT INT TERM

  export OLLAMA_HOST="127.0.0.1:${LOCAL_PORT}"
  export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH}"
  export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS}"
  export OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-none}"
  export OPENAI_FREQUENCY_PENALTY="${OPENAI_FREQUENCY_PENALTY:-0.0}"
  export MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-3000}"
  export MAX_ITEMS_PER_BATCH="${MAX_ITEMS_PER_BATCH:-5}"

  log "Running Sensemaking E2E pipeline against ${model_name} at ${OLLAMA_HOST}..."
  bash src/test_sensemaking_e2e.sh "" "" "$qualtrics_csv" "$working_dir" "$model_name" "$@"
}

COMMAND="${1:-}"
if [[ -z "$COMMAND" ]]; then
  usage
  exit 1
fi
shift

case "$COMMAND" in
  up|start)
    cmd_up "$@"
    ;;
  tunnel)
    start_tunnel "$@"
    ;;
  run-e2e)
    cmd_run_e2e "$@"
    ;;
  status)
    cmd_status "$@"
    ;;
  monitor|watch)
    cmd_monitor "$@"
    ;;
  logs)
    cmd_logs "$@"
    ;;
  stop)
    cmd_stop "$@"
    ;;
  delete|down)
    cmd_delete "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage
    exit 1
    ;;
esac
