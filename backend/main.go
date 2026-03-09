package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════

type GenerateRequest struct {
	Prompt   string `json:"prompt"`
	Template string `json:"template"`
	Context  string `json:"context"`
	UseRAG   bool   `json:"use_rag"`
	RAGLang  string `json:"rag_language,omitempty"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Messages []ChatMessage `json:"messages"`
	Context  string        `json:"context"`
	UseRAG   bool          `json:"use_rag"`
	RAGLang  string        `json:"rag_language,omitempty"`
}

type OllamaRequest struct {
	Model    string         `json:"model"`
	Messages []OllamaMsg    `json:"messages"`
	Stream   bool           `json:"stream"`
	Options  map[string]any `json:"options,omitempty"`
}

type OllamaMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OllamaResponse struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
	Done bool `json:"done"`
}

type PreviewFile struct {
	Name    string `json:"name"`
	Content string `json:"content"`
}

type SnippetRequest struct {
	Title    string   `json:"title"`
	Code     string   `json:"code"`
	Language string   `json:"language"`
	Tags     []string `json:"tags"`
	FilePath string   `json:"file_path,omitempty"`
}

type ScanRequest struct {
	Path        string `json:"path"`
	MaxFileSize int64  `json:"max_file_size,omitempty"`
}

type SearchRequest struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results,omitempty"`
	Language   string `json:"language,omitempty"`
}

// ═══════════════════════════════════════════════════════════════
// Globals
// ═══════════════════════════════════════════════════════════════

var (
	previews    = make(map[string]string)
	previewMu   sync.RWMutex
	previewID   int
	previewIDMu sync.Mutex

	ragStore   *RAGStore
	httpClient = &http.Client{Timeout: 5 * time.Minute}
)

func nextPreviewID() string {
	previewIDMu.Lock()
	defer previewIDMu.Unlock()
	previewID++
	return fmt.Sprintf("p%d", previewID)
}

func ollamaURL() string {
	u := os.Getenv("OLLAMA_URL")
	if u == "" {
		return "http://localhost:11434"
	}
	return u
}

func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

// ═══════════════════════════════════════════════════════════════
// Generate Handler — RAG-augmented
// ═══════════════════════════════════════════════════════════════

func handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	systemPrompt := buildSystemPrompt(req.Template)
	userPrompt := req.Prompt

	// ─── RAG Retrieval ─────────────────────────────
	if req.UseRAG {
		ragCtx := ragStore.Retrieve(req.Prompt, 5, req.RAGLang)
		if ragCtx.Rendered != "" {
			systemPrompt += "\n\n" + ragCtx.Rendered
		}
	}

	if req.Context != "" {
		userPrompt = fmt.Sprintf("Existing code:\n```\n%s\n```\n\nRequest: %s", req.Context, req.Prompt)
	}

	ollamaReq := OllamaRequest{
		Model: "qwen2.5-coder:0.5b",
		Messages: []OllamaMsg{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
		Stream: true,
		Options: map[string]any{
			"temperature": 0.3,
			"num_predict": 4096,
		},
	}

	streamOllama(w, ollamaReq)
}

// ═══════════════════════════════════════════════════════════════
// Chat Handler — RAG-augmented
// ═══════════════════════════════════════════════════════════════

func handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	systemContent := "You are a helpful coding assistant. You help with HTML, Tailwind CSS, JavaScript, Python, and Go. Keep answers concise and include code examples when appropriate. When generating full files, wrap code in appropriate markdown code blocks."

	// ─── RAG Retrieval ─────────────────────────────
	if req.UseRAG && len(req.Messages) > 0 {
		lastMsg := req.Messages[len(req.Messages)-1].Content
		ragCtx := ragStore.Retrieve(lastMsg, 3, req.RAGLang)
		if ragCtx.Rendered != "" {
			systemContent += "\n\n" + ragCtx.Rendered
		}
	}

	msgs := []OllamaMsg{
		{Role: "system", Content: systemContent},
	}

	if req.Context != "" {
		msgs = append(msgs, OllamaMsg{
			Role:    "system",
			Content: fmt.Sprintf("Current code context:\n```\n%s\n```", req.Context),
		})
	}

	for _, m := range req.Messages {
		msgs = append(msgs, OllamaMsg{Role: m.Role, Content: m.Content})
	}

	ollamaReq := OllamaRequest{
		Model:    "qwen2.5-coder:0.5b",
		Messages: msgs,
		Stream:   true,
		Options: map[string]any{
			"temperature": 0.4,
			"num_predict": 2048,
		},
	}

	streamOllama(w, ollamaReq)
}

// ═══════════════════════════════════════════════════════════════
// Snippet CRUD Handlers
// ═══════════════════════════════════════════════════════════════

func handleSnippetsList(w http.ResponseWriter, r *http.Request) {
	lang := r.URL.Query().Get("language")
	tag := r.URL.Query().Get("tag")
	snippets := ragStore.List(lang, tag)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"snippets": snippets,
		"total":    len(snippets),
	})
}

func handleSnippetCreate(w http.ResponseWriter, r *http.Request) {
	var req SnippetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Code == "" {
		http.Error(w, "code is required", http.StatusBadRequest)
		return
	}
	if req.Title == "" {
		req.Title = "Untitled Snippet"
	}

	snippet := ragStore.Add(&Snippet{
		Title:    req.Title,
		Code:     req.Code,
		Language: req.Language,
		Tags:     req.Tags,
		FilePath: req.FilePath,
	})

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(snippet)
}

func handleSnippetUpdate(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if id == "" {
		http.Error(w, "id required", http.StatusBadRequest)
		return
	}

	var req SnippetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	snippet, ok := ragStore.Update(id, req.Title, req.Code, req.Language, req.Tags)
	if !ok {
		http.Error(w, "snippet not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(snippet)
}

func handleSnippetDelete(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if id == "" {
		http.Error(w, "id required", http.StatusBadRequest)
		return
	}

	if !ragStore.Delete(id) {
		http.Error(w, "snippet not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "id": id})
}

// ═══════════════════════════════════════════════════════════════
// RAG Search & Scan Handlers
// ═══════════════════════════════════════════════════════════════

func handleRAGSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.MaxResults <= 0 {
		req.MaxResults = 5
	}

	results := ragStore.Search(req.Query, req.MaxResults, req.Language)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"query":   req.Query,
		"results": results,
		"total":   len(results),
	})
}

func handleRAGScan(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req ScanRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Path == "" {
		req.Path = "/workspace"
	}

	if _, err := os.Stat(req.Path); os.IsNotExist(err) {
		http.Error(w, fmt.Sprintf("directory not found: %s", req.Path), http.StatusBadRequest)
		return
	}

	log.Printf("📂 RAG scanning directory: %s", req.Path)
	count, err := ragStore.ScanDirectory(req.Path, req.MaxFileSize)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	ragStore.mu.RLock()
	total := len(ragStore.snippets)
	ragStore.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":   "scanned",
		"path":     req.Path,
		"imported": count,
		"total":    total,
	})
}

func handleRAGStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ragStore.Stats())
}

func handleRAGClear(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	ragStore.mu.Lock()
	ragStore.snippets = make(map[string]*Snippet)
	ragStore.invertedIndex = make(map[string]map[string]bool)
	ragStore.docFreq = make(map[string]int)
	ragStore.save()
	ragStore.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "cleared"})
}

// ═══════════════════════════════════════════════════════════════
// Preview & Health
// ═══════════════════════════════════════════════════════════════

func handlePreviewSave(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var file PreviewFile
	if err := json.NewDecoder(r.Body).Decode(&file); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	id := nextPreviewID()
	previewMu.Lock()
	previews[id] = file.Content
	previewMu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"id": id, "url": "/preview/" + id})
}

func handlePreviewServe(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/preview/")
	previewMu.RLock()
	content, ok := previews[id]
	previewMu.RUnlock()

	if !ok {
		http.Error(w, "Preview not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprint(w, content)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	resp, err := http.Get(ollamaURL() + "/api/tags")
	ollamaOK := err == nil && resp.StatusCode == 200
	if resp != nil {
		resp.Body.Close()
	}

	ragStore.mu.RLock()
	ragCount := len(ragStore.snippets)
	ragStore.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":       "ok",
		"ollama":       ollamaOK,
		"rag_snippets": ragCount,
		"time":         time.Now().Format(time.RFC3339),
	})
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	resp, err := http.Get(ollamaURL() + "/api/tags")
	if err != nil {
		http.Error(w, "Ollama unreachable", http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	io.Copy(w, resp.Body)
}

// ═══════════════════════════════════════════════════════════════
// Ollama Streaming
// ═══════════════════════════════════════════════════════════════

func streamOllama(w http.ResponseWriter, req OllamaRequest) {
	body, _ := json.Marshal(req)

	resp, err := httpClient.Post(ollamaURL()+"/api/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		http.Error(w, "Ollama connection failed: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	decoder := json.NewDecoder(resp.Body)
	for {
		var chunk OllamaResponse
		if err := decoder.Decode(&chunk); err != nil {
			break
		}

		data, _ := json.Marshal(map[string]any{
			"content": chunk.Message.Content,
			"done":    chunk.Done,
		})
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()

		if chunk.Done {
			break
		}
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// ═══════════════════════════════════════════════════════════════
// Template Prompts
// ═══════════════════════════════════════════════════════════════

func buildSystemPrompt(template string) string {
	base := `You are an expert code generator. Return ONLY clean, working code without any explanations or markdown formatting. Do not wrap code in backticks or code blocks. Output raw code only.`

	switch template {
	case "html":
		return base + "\nGenerate a complete, single-file HTML page. Include all CSS in a <style> tag and all JavaScript in a <script> tag. Use modern HTML5 semantic elements. Make it responsive and visually polished."
	case "tailwind":
		return base + "\nGenerate a complete HTML page using Tailwind CSS via CDN (<script src=\"https://cdn.tailwindcss.com\"></script>). Use Tailwind utility classes extensively. Include interactive JavaScript where appropriate. Make it responsive and modern."
	case "python-crud":
		return base + "\nGenerate a Python Flask CRUD API. Include all necessary imports. Use flask and flask-cors. Define a simple in-memory data store (list of dicts). Include GET /items (list), GET /items/<id>, POST /items, PUT /items/<id>, DELETE /items/<id>. Include proper error handling, JSON responses, and CORS. Add if __name__ == '__main__': app.run(debug=True, port=5000)"
	case "python-frontend":
		return base + "\nGenerate a complete HTML page with Tailwind CSS as a frontend for a Flask CRUD API at http://localhost:5000. Include full CRUD UI with list view, create form, edit, delete. Use fetch() for API calls."
	case "go-api":
		return base + "\nGenerate a Go HTTP API using only the standard library. Define a resource struct, in-memory storage with mutex, and CRUD routes: GET/POST/PUT/DELETE on /api/items. Include CORS middleware and JSON handling."
	default:
		return base + "\nGenerate clean, production-ready code based on the user's request."
	}
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Initialize RAG store
	ragDataPath := os.Getenv("RAG_DATA_PATH")
	if ragDataPath == "" {
		ragDataPath = "/data/rag/snippets.json"
	}
	ragStore = NewRAGStore(ragDataPath)
	log.Printf("📚 RAG store: %d snippets loaded", len(ragStore.snippets))

	// Auto-scan workspace on startup
	wsPath := os.Getenv("WORKSPACE_PATH")
	if wsPath == "" {
		wsPath = "/workspace"
	}
	if info, err := os.Stat(wsPath); err == nil && info.IsDir() {
		log.Printf("📂 Auto-scanning: %s", wsPath)
		count, _ := ragStore.ScanDirectory(wsPath, 50*1024)
		log.Printf("📂 Imported %d files", count)
	}

	mux := http.NewServeMux()

	// Core API
	mux.HandleFunc("/api/generate", corsMiddleware(handleGenerate))
	mux.HandleFunc("/api/chat", corsMiddleware(handleChat))
	mux.HandleFunc("/api/health", corsMiddleware(handleHealth))
	mux.HandleFunc("/api/models", corsMiddleware(handleModels))
	mux.HandleFunc("/api/preview", corsMiddleware(handlePreviewSave))

	// Snippet CRUD (method routing)
	mux.HandleFunc("/api/snippets", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "GET":
			handleSnippetsList(w, r)
		case "POST":
			handleSnippetCreate(w, r)
		case "PUT":
			handleSnippetUpdate(w, r)
		case "DELETE":
			handleSnippetDelete(w, r)
		case "OPTIONS":
			w.WriteHeader(http.StatusOK)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}))

	// RAG operations
	mux.HandleFunc("/api/rag/search", corsMiddleware(handleRAGSearch))
	mux.HandleFunc("/api/rag/scan", corsMiddleware(handleRAGScan))
	mux.HandleFunc("/api/rag/stats", corsMiddleware(handleRAGStats))
	mux.HandleFunc("/api/rag/clear", corsMiddleware(handleRAGClear))

	// Preview
	mux.HandleFunc("/preview/", handlePreviewServe)

	log.Printf("🚀 LocalCoder on :%s | Ollama: %s | RAG: %s", port, ollamaURL(), ragDataPath)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}
