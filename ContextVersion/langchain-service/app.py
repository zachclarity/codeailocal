"""
LangChain RAG Service for Zactonics AI Spring Forge
====================================================
Provides context-enriched code generation by retrieving relevant Java Spring
best-practice snippets from ChromaDB and injecting them into the LLM prompt.
"""

import json
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# -- Configuration -------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_BASE_URL = f"http://{CHROMA_HOST}:{CHROMA_PORT}"
COLLECTION_NAME = "spring_best_practices"
SNIPPETS_PATH = Path("/app/snippets/spring_best_practices.json")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "default_tenant")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "default_database")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("rag")


# -- ChromaDB v2 REST Client --------------------------------------------------
# ChromaDB v2 API uses /api/v2/tenants/{tenant}/databases/{database}/...
# System endpoints (heartbeat, version) remain at /api/v2/

class ChromaRESTClient:
    """Lightweight ChromaDB v2 HTTP client using only httpx."""

    def __init__(self, base_url: str, tenant: str = "default_tenant",
                 database: str = "default_database", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.tenant = tenant
        self.database = database
        self.timeout = timeout
        self.collection_id: Optional[str] = None
        # v2 collection prefix
        self._db_prefix = f"/api/v2/tenants/{tenant}/databases/{database}"

    def _req(self, method: str, path: str, **kwargs) -> httpx.Response:
        kwargs.setdefault("timeout", self.timeout)
        url = f"{self.base_url}{path}"
        logger.info("ChromaDB >> %s %s", method, path)
        resp = httpx.request(method, url, **kwargs)
        if resp.status_code >= 400:
            logger.error("ChromaDB << %d: %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
        logger.info("ChromaDB << %d OK", resp.status_code)
        return resp

    def heartbeat(self) -> dict:
        """System endpoint — no tenant/database prefix."""
        return self._req("GET", "/api/v2/heartbeat").json()

    def version(self) -> str:
        """Get ChromaDB server version."""
        return self._req("GET", "/api/v2/version").json()

    def get_or_create_collection(self, name: str, metadata: Optional[dict] = None) -> str:
        """Create or get a collection. Returns collection ID."""
        body: dict = {"name": name, "get_or_create": True}
        if metadata:
            body["metadata"] = metadata
        data = self._req("POST", f"{self._db_prefix}/collections", json=body).json()
        self.collection_id = data["id"]
        logger.info("Collection '%s' id=%s", name, self.collection_id)
        return self.collection_id

    def count(self) -> int:
        if not self.collection_id:
            return 0
        data = self._req("GET", f"{self._db_prefix}/collections/{self.collection_id}/count").json()
        return data if isinstance(data, int) else 0

    def add(self, ids: list, documents: list, metadatas: list, embeddings: list):
        self._req("POST", f"{self._db_prefix}/collections/{self.collection_id}/add", json={
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings,
        })

    def query(self, query_embeddings: list, n_results: int = 3,
              where: Optional[dict] = None, include: Optional[list] = None) -> dict:
        body: dict = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include or ["documents", "metadatas", "distances"],
        }
        if where:
            body["where"] = where
        return self._req("POST", f"{self._db_prefix}/collections/{self.collection_id}/query", json=body).json()

    def get_all(self, include: Optional[list] = None) -> dict:
        body = {"include": include or ["metadatas"]}
        return self._req("POST", f"{self._db_prefix}/collections/{self.collection_id}/get", json=body).json()

    def delete_collection(self, name: str):
        self._req("DELETE", f"{self._db_prefix}/collections/{name}")
        self.collection_id = None


# -- Global State --------------------------------------------------------------

chroma: Optional[ChromaRESTClient] = None


# -- Helper Functions ----------------------------------------------------------

def wait_for_chroma(max_retries: int = 30, delay: float = 2.0) -> ChromaRESTClient:
    client = ChromaRESTClient(CHROMA_BASE_URL, tenant=CHROMA_TENANT, database=CHROMA_DATABASE)
    for attempt in range(max_retries):
        try:
            hb = client.heartbeat()
            logger.info("ChromaDB is ready (attempt %d) heartbeat=%s", attempt + 1, hb)
            try:
                ver = client.version()
                logger.info("ChromaDB version: %s", ver)
            except Exception:
                pass
            return client
        except Exception as e:
            logger.info("Waiting for ChromaDB... (%d/%d) error=%s", attempt + 1, max_retries, e)
            time.sleep(delay)
    raise RuntimeError("ChromaDB not reachable after retries")


def wait_for_ollama(max_retries: int = 30, delay: float = 3.0):
    for attempt in range(max_retries):
        try:
            r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                logger.info("Ollama is ready (attempt %d)", attempt + 1)
                return
        except Exception:
            pass
        logger.info("Waiting for Ollama... (%d/%d)", attempt + 1, max_retries)
        time.sleep(delay)
    raise RuntimeError("Ollama not reachable after retries")


def get_embedding(text: str) -> list:
    logger.info("Getting embedding for %d chars...", len(text))
    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=120,
    )
    if resp.status_code != 200:
        logger.error("Embedding failed %d: %s", resp.status_code, resp.text[:300])
    resp.raise_for_status()
    data = resp.json()
    emb = data["embeddings"][0]
    logger.info("Got embedding: %d dimensions", len(emb))
    return emb


def seed_snippets(client: ChromaRESTClient):
    existing = client.count()
    if existing > 0:
        logger.info("Collection already has %d documents, skipping seed.", existing)
        return

    if not SNIPPETS_PATH.exists():
        logger.warning("Snippets file not found at %s", SNIPPETS_PATH)
        return

    snippets = json.loads(SNIPPETS_PATH.read_text())
    logger.info("Seeding %d snippets into ChromaDB...", len(snippets))

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for i, s in enumerate(snippets):
        doc_text = f"{s['title']}\n{s['description']}\n\n{s['code']}"
        ids.append(s["id"])
        documents.append(doc_text)
        metadatas.append({
            "category": s["category"],
            "title": s["title"],
            "description": s["description"],
        })
        logger.info("Embedding snippet %d/%d: %s", i + 1, len(snippets), s["id"])
        emb = get_embedding(doc_text[:2000])
        embeddings.append(emb)

    client.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    logger.info("Seeded %d snippets successfully", len(snippets))


# -- Lifespan ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chroma

    logger.info("Starting RAG service...")
    wait_for_ollama()

    logger.info("Ensuring embedding model '%s' is available...", EMBEDDING_MODEL)
    try:
        httpx.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": EMBEDDING_MODEL, "stream": False},
            timeout=600,
        )
        logger.info("Embedding model ready")
    except Exception as e:
        logger.warning("Could not pull embedding model: %s", e)

    chroma = wait_for_chroma()
    chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    seed_snippets(chroma)
    logger.info("=== RAG SERVICE READY ===")

    yield

    logger.info("Shutting down RAG service")


# -- FastAPI App ---------------------------------------------------------------

app = FastAPI(
    title="Spring Forge RAG API",
    description="RAG-powered Java Spring code generation",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Request / Response Models -------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User's code generation prompt")
    model: str = Field(default="qwen2.5-coder:0.5b", description="Ollama model name")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of snippets to retrieve")
    stream: bool = Field(default=True, description="Stream the response")
    category: Optional[str] = Field(default=None, description="Filter snippets by category")


class SnippetResult(BaseModel):
    id: str
    title: str
    category: str
    description: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SnippetResult]


class HealthResponse(BaseModel):
    status: str
    chroma_connected: bool
    ollama_connected: bool
    snippet_count: int


# -- Endpoints -----------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    chroma_ok = False
    ollama_ok = False
    count = 0

    try:
        if chroma:
            chroma.heartbeat()
            chroma_ok = True
            count = chroma.count()
    except Exception as e:
        logger.warning("Health check ChromaDB failed: %s", e)

    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception as e:
        logger.warning("Health check Ollama failed: %s", e)

    return HealthResponse(
        status="ok" if (chroma_ok and ollama_ok) else "degraded",
        chroma_connected=chroma_ok,
        ollama_connected=ollama_ok,
        snippet_count=count,
    )


@app.post("/search", response_model=SearchResponse)
async def search_snippets(prompt: str, top_k: int = 5, category: Optional[str] = None):
    if chroma is None:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")

    query_embedding = get_embedding(prompt)
    where_filter = {"category": category} if category else None

    results = chroma.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["metadatas", "distances"],
    )

    snippets = []
    if results.get("ids") and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            distance = results["distances"][0][i] if results.get("distances") else 0
            snippets.append(SnippetResult(
                id=doc_id,
                title=meta.get("title", ""),
                category=meta.get("category", ""),
                description=meta.get("description", ""),
                score=round(1 - distance, 4),
            ))

    return SearchResponse(query=prompt, results=snippets)


@app.post("/generate")
async def generate_code(req: GenerateRequest):
    if chroma is None:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")

    logger.info("=== GENERATE === model=%s category=%s top_k=%d", req.model, req.category, req.top_k)
    logger.info("Prompt: %.200s", req.prompt)

    # Step 1: Embed the user prompt
    try:
        query_embedding = get_embedding(req.prompt)
        logger.info("Step 1 OK: embedding %d dims", len(query_embedding))
    except Exception as e:
        logger.error("Step 1 FAILED: %s", e)
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}")

    # Step 2: Search ChromaDB
    try:
        where_filter = {"category": req.category} if req.category else None
        results = chroma.query(
            query_embeddings=[query_embedding],
            n_results=req.top_k,
            where=where_filter,
            include=["documents", "metadatas"],
        )
        doc_count = len(results.get("documents", [[]])[0])
        logger.info("Step 2 OK: %d snippets retrieved", doc_count)
    except Exception as e:
        logger.error("Step 2 FAILED: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=502, detail=f"ChromaDB query failed: {e}")

    # Build context
    context_blocks = []
    if results.get("documents") and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            title = meta.get("title", "Snippet")
            logger.info("  Context %d: %s", i + 1, title)
            context_blocks.append(
                f"=== Best Practice Reference: {title} ===\n"
                f"Category: {meta.get('category', 'general')}\n"
                f"{meta.get('description', '')}\n\n"
                f"{doc}\n"
            )

    context = "\n".join(context_blocks)
    logger.info("Step 2: context = %d chars from %d snippets", len(context), len(context_blocks))

    # Step 3: Build system prompt
    system_prompt = f"""You are a senior Java Spring Boot developer. You ONLY output clean, production-ready Java code.

REFERENCE CONTEXT - Use these best-practice examples as guidance for style, conventions, and patterns:

{context}

RULES:
- Output ONLY the Java source code, no markdown fences, no explanations outside the code.
- Use Java 17+ features where appropriate (records, sealed classes, pattern matching).
- Follow Spring Boot 3.x conventions.
- Include proper imports.
- Use Lombok where appropriate (@RequiredArgsConstructor, @Slf4j, @Data, @Builder).
- Add detailed Javadoc and inline // Best Practice: comments explaining WHY each decision was made.
- Follow the patterns shown in the reference context above.
- Apply SOLID principles and clean code conventions.

IMPORTANT: Include "// Best Practice:" comments throughout the code to explain design decisions,
just like the reference examples. These comments teach the developer WHY the code is written this way.

Do NOT wrap code in ```java or any markdown. Output raw Java code only."""

    logger.info("Step 3: system prompt = %d chars", len(system_prompt))

    # Step 4: Stream from Ollama
    if req.stream:
        return StreamingResponse(
            _stream_ollama(req.model, system_prompt, req.prompt),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    else:
        return await _generate_ollama(req.model, system_prompt, req.prompt)


async def _stream_ollama(model: str, system_prompt: str, user_prompt: str):
    """Stream tokens from Ollama as SSE events. All errors are yielded, never raised."""
    token_count = 0
    try:
        logger.info("Step 4: POST /api/chat stream=true model=%s", model)

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)
        ) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    error_msg = f"Ollama returned {response.status_code}: {body.decode()[:500]}"
                    logger.error("Step 4 FAILED: %s", error_msg)
                    yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                logger.info("Step 4: Ollama streaming (status 200)")

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content")
                        if content is not None and content != "":
                            token_count += 1
                            sse_event = {
                                "type": "content_block_delta",
                                "delta": {"text": content},
                            }
                            yield f"data: {json.dumps(sse_event)}\n\n"

                        if data.get("done"):
                            logger.info("Step 4 DONE: %d tokens streamed", token_count)
                            yield "data: [DONE]\n\n"
                            return
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON from Ollama: %.200s", line)
                        continue

        if token_count == 0:
            logger.warning("Step 4: stream ended with 0 tokens")
            yield f"data: {json.dumps({'type': 'error', 'error': 'Ollama returned 0 tokens. Is the model loaded?'})}\n\n"
        yield "data: [DONE]\n\n"

    except httpx.ConnectError as e:
        msg = f"Cannot connect to Ollama at {OLLAMA_BASE_URL}: {e}"
        logger.error("Step 4 FAILED: %s", msg)
        yield f"data: {json.dumps({'type': 'error', 'error': msg})}\n\n"
        yield "data: [DONE]\n\n"
    except httpx.ReadTimeout:
        msg = f"Ollama read timeout after {token_count} tokens"
        logger.error("Step 4 FAILED: %s", msg)
        yield f"data: {json.dumps({'type': 'error', 'error': msg})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        msg = f"Stream error: {type(e).__name__}: {e}"
        logger.error("Step 4 FAILED: %s\n%s", msg, traceback.format_exc())
        yield f"data: {json.dumps({'type': 'error', 'error': msg})}\n\n"
        yield "data: [DONE]\n\n"


async def _generate_ollama(model: str, system_prompt: str, user_prompt: str) -> dict:
    logger.info("Calling Ollama /api/chat non-stream model=%s", model)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)
    ) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        code = data.get("message", {}).get("content", "")
        logger.info("Non-stream response: %d chars", len(code))
        return {
            "code": code,
            "model": model,
            "context_snippets_used": chroma.count() if chroma else 0,
        }


@app.get("/snippets")
async def list_snippets():
    if chroma is None:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    results = chroma.get_all(include=["metadatas"])
    items = []
    if results.get("ids"):
        for i in range(len(results["ids"])):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            items.append({"id": results["ids"][i], **meta})
    return {"count": len(items), "snippets": items}


@app.post("/snippets/reload")
async def reload_snippets():
    if chroma is None:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    seed_snippets(chroma)
    return {"status": "reloaded", "count": chroma.count()}


# -- Debug endpoint ------------------------------------------------------------

@app.post("/generate/debug")
async def generate_debug(req: GenerateRequest):
    """Returns diagnostic info for each pipeline step."""
    if chroma is None:
        return JSONResponse({"error": "ChromaDB not initialized"}, status_code=503)

    steps = {}

    # Check ChromaDB
    try:
        hb = chroma.heartbeat()
        cnt = chroma.count()
        steps["chromadb"] = {"status": "ok", "heartbeat": hb, "snippet_count": cnt}
    except Exception as e:
        steps["chromadb"] = {"status": "error", "error": str(e)}
        return {"steps": steps}

    # Check embedding
    try:
        emb = get_embedding(req.prompt[:200])
        steps["embedding"] = {"status": "ok", "dimensions": len(emb)}
    except Exception as e:
        steps["embedding"] = {"status": "error", "error": str(e)}
        return {"steps": steps}

    # Check query
    try:
        where_filter = {"category": req.category} if req.category else None
        results = chroma.query(
            query_embeddings=[emb], n_results=req.top_k,
            where=where_filter, include=["metadatas", "distances"],
        )
        found = []
        if results.get("ids") and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                dist = results["distances"][0][i] if results.get("distances") else 0
                found.append({"id": doc_id, "title": meta.get("title", ""), "similarity": round(1 - dist, 4)})
        steps["query"] = {"status": "ok", "results": found}
    except Exception as e:
        steps["query"] = {"status": "error", "error": str(e)}
        return {"steps": steps}

    # Check Ollama model
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        steps["ollama"] = {
            "status": "ok",
            "models_loaded": models,
            "requested_model": req.model,
            "model_available": any(req.model in m for m in models),
        }
    except Exception as e:
        steps["ollama"] = {"status": "error", "error": str(e)}

    return {"steps": steps}
