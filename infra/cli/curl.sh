#!/bin/sh
set -eu

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "usage: $0 <request.json>" >&2
    exit 2
fi

: "${OPENAI_API_URL:?set OPENAI_API_URL}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"

exec curl --fail-with-body --silent --show-error \
    -X POST "$OPENAI_API_URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d "@$1"
