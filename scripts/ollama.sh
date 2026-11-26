#!/usr/bin/env bash
set -euo pipefail

: "${OLLAMA_ROOT:=/goinfre/$USER/ollama}"
: "${OLLAMA_MODEL:=qwen3:0.6b}"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$OLLAMA_ROOT/models"
export OLLAMA_TMPDIR="$OLLAMA_ROOT/tmp"

# Resolve symlinks so a home-directory alias cannot hide the storage location.
OLLAMA_ROOT=$(realpath -m "$OLLAMA_ROOT")
case "$OLLAMA_ROOT/" in
    "$HOME/"*|/home/*)
        echo 'OLLAMA_ROOT must be outside /home.' >&2
        exit 1
        ;;
esac
export OLLAMA_ROOT
export OLLAMA_MODELS="$OLLAMA_ROOT/models"
export OLLAMA_TMPDIR="$OLLAMA_ROOT/tmp"
for tool in curl tar zstd; do
    command -v "$tool" >/dev/null || { echo "Required command missing: $tool" >&2; exit 1; }
done
mkdir -p "$OLLAMA_ROOT" "$OLLAMA_MODELS" "$OLLAMA_TMPDIR" "$OLLAMA_ROOT/state"

if [ ! -x "$OLLAMA_ROOT/bin/ollama" ]; then
    case "$(uname -m)" in
        x86_64) arch=amd64 ;;
        aarch64|arm64) arch=arm64 ;;
        *) echo 'Unsupported architecture.' >&2; exit 1 ;;
    esac
    staging=$(mktemp -d "$OLLAMA_ROOT/.install.XXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    echo "Installing Ollama in $OLLAMA_ROOT"
    curl --fail --location --retry 3 \
        "https://ollama.com/download/ollama-linux-$arch.tar.zst" \
        -o "$staging/ollama.tar.zst"
    mkdir "$staging/unpacked"
    tar --zstd -xf "$staging/ollama.tar.zst" -C "$staging/unpacked"
    # Publish the executable last so interrupted installs are retried.
    cp -a "$staging/unpacked/lib" "$OLLAMA_ROOT/"
    mkdir -p "$OLLAMA_ROOT/bin"
    cp -a "$staging/unpacked/bin/." "$OLLAMA_ROOT/bin/"
    rm -rf -- "$staging"
    trap - EXIT
fi

# Ollama stores keys and history in ~/.ollama. Keep their contents in goinfre.
if [ ! -e "$HOME/.ollama" ] && [ ! -L "$HOME/.ollama" ]; then
    ln -s "$OLLAMA_ROOT/state" "$HOME/.ollama"
elif [ -L "$HOME/.ollama" ] && [ "$(realpath -m "$HOME/.ollama")" = "$OLLAMA_ROOT/state" ]; then
    :
else
    echo "Existing $HOME/.ollama does not point to $OLLAMA_ROOT/state; move it there and replace it with a symlink first." >&2
    exit 1
fi

api() {
    curl --noproxy '*' --fail --silent --show-error --max-time 2 \
        "http://$OLLAMA_HOST/$1"
}
if ! api api/version >/dev/null 2>&1; then
    echo "Starting Ollama (log: $OLLAMA_ROOT/ollama.log)"
    nohup "$OLLAMA_ROOT/bin/ollama" serve >"$OLLAMA_ROOT/ollama.log" 2>&1 </dev/null &
    server_pid=$!
    ready=false
    for ((attempt=0; attempt<60; attempt++)); do
        if api api/version >/dev/null 2>&1; then ready=true; break; fi
        if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
        sleep 1
    done
    if [ "$ready" != true ]; then
        echo "Ollama failed to start. See $OLLAMA_ROOT/ollama.log" >&2
        exit 1
    fi
fi

if ! "$OLLAMA_ROOT/bin/ollama" show "$OLLAMA_MODEL" >/dev/null 2>&1; then
    "$OLLAMA_ROOT/bin/ollama" pull "$OLLAMA_MODEL"
fi
# Load the model into memory and keep it available for subsequent requests.
curl --noproxy '*' --fail --silent --show-error --max-time 300 \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$OLLAMA_MODEL\",\"keep_alive\":-1,\"stream\":false}" \
    "http://$OLLAMA_HOST/api/generate"
printf '\n%s is ready at http://%s (models: %s)\n' "$OLLAMA_MODEL" "$OLLAMA_HOST" "$OLLAMA_MODELS"
