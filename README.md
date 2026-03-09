# LocalCoder — AI Code Studio with RAG

A local, self-hosted AI coding environment with **Retrieval-Augmented Generation** from your own code. Think Cursor, but 100% local — no API keys, no cloud.

## Stack

| Service | Port | Description |
|---------|------|-------------|
| **Frontend** | `3000` | Code editor, AI chat, snippet manager, live preview |
| **Backend** | `8080` | Go API — RAG engine, Ollama proxy, preview server |
| **Ollama** | `11434` | Qwen2.5-Coder:0.5b running locally |

## Quick Start

```bash
cd localcoder

# Point to your code directory (for RAG indexing)
export WORKSPACE_DIR=~/projects/my-app
# Or edit .env.example → .env

docker compose up --build
# First run pulls the ~400MB model — wait for "Model ready!"

open http://localhost:3000
```

On startup the backend automatically scans `WORKSPACE_DIR` and indexes all code files into the RAG store.

## RAG System

The RAG (Retrieval-Augmented Generation) system lets the AI reference your existing code when generating new code.

### How It Works

1. **Index** — Code files are tokenized and stored with TF-IDF vectors
2. **Retrieve** — When you generate code or chat, the system searches for relevant snippets
3. **Augment** — Matching snippets are injected into the LLM's system prompt as context
4. **Generate** — The model produces code that follows your existing patterns and conventions

### Adding Code to RAG

There are several ways to build your RAG knowledge base:

- **Auto-scan on startup** — Any code in `WORKSPACE_DIR` is indexed automatically
- **Manual scan** — Use the Manage tab to scan additional directories
- **Save from editor** — Click the save icon to store current editor code as a snippet
- **Add manually** — Use the "Add Snippet" button to paste code with custom tags
- **Chat/Generate** — Toggle "Use RAG" to include snippets in AI context

### Supported File Types

`.html` `.css` `.js` `.ts` `.jsx` `.tsx` `.py` `.go` `.rs` `.java` `.rb` `.php` `.sql` `.sh` `.yml` `.json` `.md` `.vue` `.svelte`

Directories like `node_modules`, `.git`, `__pycache__`, `vendor`, `dist` are auto-skipped.

### Search

The snippet search uses TF-IDF cosine similarity with:
- Inverted index for fast candidate retrieval
- camelCase / snake_case token splitting
- Tag and title boosting
- Substring matching for partial terms
- Language filtering

### API Endpoints

```
POST /api/rag/search   — Search snippets  { query, max_results, language }
POST /api/rag/scan     — Scan directory    { path, max_file_size }
GET  /api/rag/stats    — Index statistics
POST /api/rag/clear    — Clear all snippets

GET    /api/snippets          — List (filter: ?language=&tag=)
POST   /api/snippets          — Create { title, code, language, tags }
PUT    /api/snippets?id=X     — Update
DELETE /api/snippets?id=X     — Delete
```

## Features

### Code Generation (Ctrl+G)
- Select template → describe → generate with streaming
- **RAG toggle** injects relevant snippets into the prompt
- Templates: HTML, Tailwind, Python CRUD, Go API

### AI Chat (Ctrl+B)
- Context-aware: sees your current editor code
- **RAG toggle** retrieves snippets matching your question
- "Apply to Editor" to paste generated code

### Snippet Panel (Ctrl+K)
- **Snippets tab** — Browse all indexed code
- **Search tab** — TF-IDF semantic search with language filter
- **Manage tab** — Stats, directory scanner, language breakdown

### Live Preview (Ctrl+Enter)
HTML preview in embedded iframe served through the backend.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+G` | Focus generate prompt |
| `Ctrl+B` | Toggle AI chat sidebar |
| `Ctrl+K` | Toggle snippets panel |
| `Ctrl+Enter` | Run preview |
| `Escape` | Close panels |

## Configuration

### Point to Your Code

```bash
# Option 1: Environment variable
WORKSPACE_DIR=~/my-project docker compose up

# Option 2: .env file
echo "WORKSPACE_DIR=~/my-project" > .env
docker compose up
```

### GPU Support

Uncomment the `deploy.resources` block in `docker-compose.yml` if you have NVIDIA GPU + container toolkit.

### Bigger Model

```bash
docker exec localcoder-ollama ollama pull qwen2.5-coder:7b
```
Then update `"qwen2.5-coder:0.5b"` → `"qwen2.5-coder:7b"` in `backend/main.go`.

## Architecture

```
┌────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│  Frontend      │────▶│  Go Backend          │────▶│  Ollama      │
│  (Nginx:3000)  │     │  (:8080)             │     │  (:11434)    │
│                │     │                      │     │              │
│ Code Editor    │     │ /api/generate  ──────│──┐  │ Qwen2.5-    │
│ AI Chat        │     │ /api/chat      ──────│──┤  │ Coder 0.5b  │
│ Snippet Panel  │     │ /api/snippets  CRUD  │  │  │              │
│ Live Preview   │     │ /api/rag/*     ──┐   │  │  └──────────────┘
│                │     │                  │   │  │
│ RAG Toggle ◉   │     │  ┌───────────────┴─┐ │  │
│                │     │  │  RAG Engine     │ │  │
└────────────────┘     │  │  TF-IDF Index   │──┘  │
                       │  │  JSON Persist   │     │
  ┌─────────────┐      │  │  Dir Scanner    │     │
  │ /workspace  │──ro──│  └─────────────────┘     │
  │ Your Code   │      └─────────────────────┘     │
  └─────────────┘
```

## Sample Workspace

The `workspace/` directory includes 3 sample files that get auto-indexed:
- `auth_middleware.py` — Flask authentication patterns
- `data_table.html` — Reusable Tailwind data table component
- `go_crud.go` — Go standard library CRUD handlers

Try generating "Create a Flask API with authentication" with RAG enabled — the model will reference your auth middleware patterns.
