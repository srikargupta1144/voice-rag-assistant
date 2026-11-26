# Cloud deployment guide

This guide describes how to run the Voice RAG Assistant in a managed container service with the hosted-LLM setting. Nothing in this repository deploys automatically; every step below is run by hand by someone with access to the cloud account.

## Why the hosted LLM in the cloud

The API container runs speech-to-text (faster-whisper small, int8), retrieval (bge-small, FAISS, BM25, MiniLM re-ranker) and text-to-speech (Piper) comfortably on 2 vCPU and 4 GB of memory. A local Llama model does not: on CPU-only cloud instances a 3B model takes tens of seconds per answer and needs a much larger, more expensive instance. The cloud deployment therefore switches the LLM to an OpenAI-compatible hosted API with one setting, while everything else stays identical to the local build:

| Setting | Local | Cloud |
| --- | --- | --- |
| `VRA_LLM__PROVIDER` | `ollama` | `openai` |
| `VRA_LLM__OPENAI_BASE_URL` | not used | `https://api.openai.com/v1` or another OpenAI-compatible endpoint (for example an Azure OpenAI v1 endpoint) |
| `VRA_LLM__OPENAI_MODEL` | not used | for example `gpt-4o-mini` |
| `VRA_LLM__API_KEY` | not used | secret, injected from the platform's secret store |
| `PINECONE_API_KEY` | unset (FAISS) | optional; when set, the vector store switches to Pinecone |

The container image is the same one `docker compose` builds locally (see the root `Dockerfile`). It bakes in the sample documents, the FAISS index and all local models, so it starts without downloads. Its health check calls `GET /health`.

## Common preparation

1. Build the image: `docker build -t voice-rag-assistant:<version> .`
2. Run it once locally with the hosted setting to confirm the key works: `docker run --rm -p 8000:8000 -e VRA_LLM__PROVIDER=openai -e VRA_LLM__API_KEY=... voice-rag-assistant:<version>`, then `curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is the late fee on the Kova card?\"}"`.
3. Decide on the vector store. With FAISS the index lives inside the image and is rebuilt by rebuilding the image. With Pinecone, set `PINECONE_API_KEY`, run `vra ingest` once (locally or as a one-off container job) to upsert the vectors, and every replica then reads the same index.

## Option A: Azure Container Apps

1. Create a resource group, an Azure Container Registry and a Container Apps environment (`az group create`, `az acr create`, `az containerapp env create`).
2. Push the image: `az acr login --name <registry>`, then `docker tag` and `docker push <registry>.azurecr.io/voice-rag-assistant:<version>`.
3. Store the LLM key as a Container Apps secret and reference it from an environment variable, for example `--secrets llm-key=<key> --env-vars VRA_LLM__PROVIDER=openai VRA_LLM__OPENAI_MODEL=gpt-4o-mini VRA_LLM__API_KEY=secretref:llm-key`.
4. Create the app with external ingress on port 8000, 2 vCPU and 4 GiB memory: `az containerapp create --name voice-rag --resource-group <rg> --environment <env> --image <registry>.azurecr.io/voice-rag-assistant:<version> --target-port 8000 --ingress external --cpu 2 --memory 4Gi --min-replicas 1 --max-replicas 3 --registry-server <registry>.azurecr.io ...`.
5. Keep `--min-replicas 1`: model warm-up at start-up takes several seconds, so scaling to zero would make the first question slow.
6. WebSockets work through Container Apps ingress without extra settings; the page at the app URL uses `wss://` automatically.
7. Open the app URL, check `/health`, and ask a question from the page.

## Option B: AWS ECS on Fargate

1. Create an ECR repository and push the image (`aws ecr create-repository`, `aws ecr get-login-password | docker login ...`, `docker push`).
2. Store the LLM key in AWS Secrets Manager.
3. Register a task definition: Fargate, 2 vCPU, 4 GB, container port 8000, environment `VRA_LLM__PROVIDER=openai` and `VRA_LLM__OPENAI_MODEL=...`, and `VRA_LLM__API_KEY` taken from the secret through the task definition's `secrets` block. The task execution role needs permission to read that secret and to pull from ECR.
4. Create an ECS service behind an Application Load Balancer with a target group health check on `/health`. Raise the ALB idle timeout to at least 120 seconds so long answers and WebSocket turns are not cut off.
5. Use at least one running task (no scale to zero) for the same warm-up reason as above, and add target tracking on CPU for scaling out.
6. Put an HTTPS listener (ACM certificate) in front so the browser can use the microphone, which requires a secure origin.

## After deployment

- Logs: every line carries `request_id=...`, and each answer logs its per-stage timings (`stt_ms`, `retrieval_ms`, `llm_first_token_ms`, `tts_ms`, `total_ms`), which can be charted with the platform's log queries.
- Timeouts: `VRA_API__REQUEST_TIMEOUT_S` (default 120) bounds each request; the load balancer timeout should be higher.
- Cost control: the hosted model is the main variable cost; `VRA_LLM__MAX_TOKENS` (default 300) caps answer length.
- The microphone only works on HTTPS origins (or localhost), so always serve the page over TLS in the cloud.
