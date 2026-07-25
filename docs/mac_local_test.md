# Launch the service

GLOO_SOCKET_IFNAME=lo0 \
VLLM_HOST_IP=127.0.0.1 \
VLLM_LOOPBACK_IP=127.0.0.1 \
HF_HUB_DISABLE_XET=1 \
uv run vllm serve Qwen/Qwen3.5-0.8B \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.3 \
  --enforce-eager \
  --skip-mm-profiling \
  --mm-processor-kwargs '{"max_pixels":1048576}'


# request template

curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
            }
          },
          {
            "type": "text",
            "text": "What animal is on the candy?"
          }
        ]
      }
    ],
    "max_tokens": 40,
    "temperature": 0
  }'
