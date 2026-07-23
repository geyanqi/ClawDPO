#!/bin/sh
set -eu

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "usage: $0 <where.sql>" >&2
    exit 2
fi

: "${DATABASE_API_URL:?set DATABASE_API_URL}"
: "${DATABASE_API_KEY:?set DATABASE_API_KEY}"

{
    printf '%s\n' "select create_time, conversation_detail" \
        "from openai_log_proxy"
    cat "$1"
} | curl --fail-with-body --silent --show-error \
    -X POST "$DATABASE_API_URL" \
    -H "Content-Type: text/plain" \
    -H "Accept: text/csv" \
    -H "Authorization: Bearer $DATABASE_API_KEY" \
    --data-binary @-
