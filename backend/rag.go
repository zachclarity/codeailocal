package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
)

// ═══════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════

type Snippet struct {
	ID        string    `json:"id"`
	Title     string    `json:"title"`
	Code      string    `json:"code"`
	Language  string    `json:"language"`
	Tags      []string  `json:"tags"`
	FilePath  string    `json:"file_path,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	// Precomputed TF vector for fast search
	tfVector map[string]float64 `json:"-"`
}

type SearchResult struct {
	Snippet Snippet `json:"snippet"`
	Score   float64 `json:"score"`
	Reason  string  `json:"reason"`
}

type RAGContext struct {
	Query    string         `json:"query"`
	Results  []SearchResult `json:"results"`
	Rendered string         `json:"rendered"`
}

// ═══════════════════════════════════════════════════════════════
// RAG Store
// ═══════════════════════════════════════════════════════════════

type RAGStore struct {
	mu       sync.RWMutex
	snippets map[string]*Snippet
	// Inverted index: token → set of snippet IDs
	invertedIndex map[string]map[string]bool
	// Document frequency: token → count of docs containing it
	docFreq  map[string]int
	savePath string
}

func NewRAGStore(savePath string) *RAGStore {
	store := &RAGStore{
		snippets:      make(map[string]*Snippet),
		invertedIndex: make(map[string]map[string]bool),
		docFreq:       make(map[string]int),
		savePath:      savePath,
	}
	store.load()
	return store
}

// ─── Snippet CRUD ──────────────────────────────────────────────

func (s *RAGStore) Add(snippet *Snippet) *Snippet {
	s.mu.Lock()
	defer s.mu.Unlock()

	if snippet.ID == "" {
		hash := sha256.Sum256([]byte(snippet.Code + snippet.Title + time.Now().String()))
		snippet.ID = fmt.Sprintf("snip_%x", hash[:8])
	}
	snippet.CreatedAt = time.Now()
	snippet.UpdatedAt = time.Now()

	if snippet.Language == "" {
		snippet.Language = detectLanguage(snippet.Code, snippet.FilePath)
	}
	if len(snippet.Tags) == 0 {
		snippet.Tags = autoTag(snippet.Code, snippet.Language)
	}

	// Compute TF vector
	snippet.tfVector = computeTF(tokenize(snippet.Code + " " + snippet.Title + " " + strings.Join(snippet.Tags, " ")))

	s.snippets[snippet.ID] = snippet
	s.indexSnippet(snippet)
	s.save()
	return snippet
}

func (s *RAGStore) Update(id string, title, code, language string, tags []string) (*Snippet, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	snip, ok := s.snippets[id]
	if !ok {
		return nil, false
	}

	// Remove old index
	s.unindexSnippet(snip)

	if title != "" {
		snip.Title = title
	}
	if code != "" {
		snip.Code = code
	}
	if language != "" {
		snip.Language = language
	}
	if tags != nil {
		snip.Tags = tags
	}
	snip.UpdatedAt = time.Now()
	snip.tfVector = computeTF(tokenize(snip.Code + " " + snip.Title + " " + strings.Join(snip.Tags, " ")))

	// Re-index
	s.indexSnippet(snip)
	s.save()
	return snip, true
}

func (s *RAGStore) Delete(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	snip, ok := s.snippets[id]
	if !ok {
		return false
	}

	s.unindexSnippet(snip)
	delete(s.snippets, id)
	s.save()
	return true
}

func (s *RAGStore) Get(id string) (*Snippet, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	snip, ok := s.snippets[id]
	return snip, ok
}

func (s *RAGStore) List(language, tag string) []*Snippet {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var results []*Snippet
	for _, snip := range s.snippets {
		if language != "" && !strings.EqualFold(snip.Language, language) {
			continue
		}
		if tag != "" {
			found := false
			for _, t := range snip.Tags {
				if strings.EqualFold(t, tag) {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		results = append(results, snip)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].UpdatedAt.After(results[j].UpdatedAt)
	})
	return results
}

func (s *RAGStore) Stats() map[string]any {
	s.mu.RLock()
	defer s.mu.RUnlock()

	langCount := make(map[string]int)
	tagCount := make(map[string]int)
	for _, snip := range s.snippets {
		langCount[snip.Language]++
		for _, t := range snip.Tags {
			tagCount[t]++
		}
	}

	return map[string]any{
		"total_snippets": len(s.snippets),
		"languages":      langCount,
		"top_tags":        topN(tagCount, 20),
		"index_tokens":    len(s.invertedIndex),
	}
}

// ─── TF-IDF Search ─────────────────────────────────────────────

func (s *RAGStore) Search(query string, maxResults int, language string) []SearchResult {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if maxResults <= 0 {
		maxResults = 5
	}

	queryTokens := tokenize(query)
	if len(queryTokens) == 0 {
		return nil
	}

	queryTF := computeTF(queryTokens)
	totalDocs := float64(len(s.snippets))
	if totalDocs == 0 {
		return nil
	}

	// Compute TF-IDF for query
	queryVec := make(map[string]float64)
	for token, tf := range queryTF {
		df := float64(s.docFreq[token])
		if df == 0 {
			continue
		}
		idf := math.Log(1 + totalDocs/df)
		queryVec[token] = tf * idf
	}

	// Find candidate snippets via inverted index
	candidates := make(map[string]bool)
	for _, token := range queryTokens {
		if ids, ok := s.invertedIndex[token]; ok {
			for id := range ids {
				candidates[id] = true
			}
		}
	}

	// Also do substring matching for longer tokens
	for _, token := range queryTokens {
		if len(token) >= 4 {
			for indexToken, ids := range s.invertedIndex {
				if strings.Contains(indexToken, token) || strings.Contains(token, indexToken) {
					for id := range ids {
						candidates[id] = true
					}
				}
			}
		}
	}

	// Score each candidate with cosine similarity
	type scored struct {
		id    string
		score float64
	}
	var scores []scored

	for id := range candidates {
		snip := s.snippets[id]
		if snip == nil {
			continue
		}
		if language != "" && !strings.EqualFold(snip.Language, language) {
			continue
		}

		// Build doc TF-IDF vector
		docVec := make(map[string]float64)
		for token, tf := range snip.tfVector {
			df := float64(s.docFreq[token])
			if df == 0 {
				continue
			}
			idf := math.Log(1 + totalDocs/df)
			docVec[token] = tf * idf
		}

		score := cosineSimilarity(queryVec, docVec)

		// Boost for tag matches
		for _, tag := range snip.Tags {
			tagLower := strings.ToLower(tag)
			for _, qt := range queryTokens {
				if tagLower == qt || strings.Contains(tagLower, qt) {
					score += 0.15
				}
			}
		}

		// Boost for title matches
		titleTokens := tokenize(snip.Title)
		for _, tt := range titleTokens {
			for _, qt := range queryTokens {
				if tt == qt {
					score += 0.1
				}
			}
		}

		if score > 0.01 {
			scores = append(scores, scored{id, score})
		}
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	var results []SearchResult
	for i, sc := range scores {
		if i >= maxResults {
			break
		}
		snip := s.snippets[sc.id]
		if snip == nil {
			continue
		}
		results = append(results, SearchResult{
			Snippet: *snip,
			Score:   sc.score,
			Reason:  fmt.Sprintf("TF-IDF score: %.3f", sc.score),
		})
	}

	return results
}

// Retrieve builds a RAG context string for injection into LLM prompts
func (s *RAGStore) Retrieve(query string, maxResults int, language string) *RAGContext {
	results := s.Search(query, maxResults, language)
	if len(results) == 0 {
		return &RAGContext{Query: query}
	}

	var parts []string
	parts = append(parts, fmt.Sprintf("## Relevant Code Snippets (from local library, %d found):\n", len(results)))

	for i, r := range results {
		parts = append(parts, fmt.Sprintf("### Snippet %d: %s [%s] (relevance: %.0f%%)",
			i+1, r.Snippet.Title, r.Snippet.Language, r.Score*100))
		if len(r.Snippet.Tags) > 0 {
			parts = append(parts, fmt.Sprintf("Tags: %s", strings.Join(r.Snippet.Tags, ", ")))
		}
		parts = append(parts, fmt.Sprintf("```%s\n%s\n```", r.Snippet.Language, r.Snippet.Code))
		parts = append(parts, "")
	}

	parts = append(parts, "---")
	parts = append(parts, "Use the above snippets as reference. Follow similar patterns, naming conventions, and coding style when generating new code. Adapt and extend these snippets to fulfill the request.")

	rendered := strings.Join(parts, "\n")

	return &RAGContext{
		Query:    query,
		Results:  results,
		Rendered: rendered,
	}
}

// ─── Indexing ──────────────────────────────────────────────────

func (s *RAGStore) indexSnippet(snip *Snippet) {
	tokens := tokenize(snip.Code + " " + snip.Title + " " + strings.Join(snip.Tags, " "))
	seen := make(map[string]bool)
	for _, token := range tokens {
		if !seen[token] {
			seen[token] = true
			if s.invertedIndex[token] == nil {
				s.invertedIndex[token] = make(map[string]bool)
			}
			s.invertedIndex[token][snip.ID] = true
			s.docFreq[token]++
		}
	}
}

func (s *RAGStore) unindexSnippet(snip *Snippet) {
	tokens := tokenize(snip.Code + " " + snip.Title + " " + strings.Join(snip.Tags, " "))
	seen := make(map[string]bool)
	for _, token := range tokens {
		if !seen[token] {
			seen[token] = true
			if ids, ok := s.invertedIndex[token]; ok {
				delete(ids, snip.ID)
				if len(ids) == 0 {
					delete(s.invertedIndex, token)
				}
			}
			s.docFreq[token]--
			if s.docFreq[token] <= 0 {
				delete(s.docFreq, token)
			}
		}
	}
}

// ─── Directory Scanner ─────────────────────────────────────────

var supportedExtensions = map[string]string{
	".html": "html",
	".htm":  "html",
	".css":  "css",
	".js":   "javascript",
	".ts":   "typescript",
	".py":   "python",
	".go":   "go",
	".rs":   "rust",
	".java": "java",
	".rb":   "ruby",
	".php":  "php",
	".sql":  "sql",
	".sh":   "bash",
	".yml":  "yaml",
	".yaml": "yaml",
	".json": "json",
	".md":   "markdown",
	".jsx":  "jsx",
	".tsx":  "tsx",
	".vue":  "vue",
	".svelte": "svelte",
}

var ignoreDirs = map[string]bool{
	"node_modules": true, ".git": true, "__pycache__": true,
	".venv": true, "venv": true, "vendor": true, "dist": true,
	"build": true, ".next": true, ".cache": true, "target": true,
}

func (s *RAGStore) ScanDirectory(dirPath string, maxFileSize int64) (int, error) {
	if maxFileSize <= 0 {
		maxFileSize = 50 * 1024 // 50KB default
	}

	count := 0
	err := filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip errors
		}

		if d.IsDir() {
			if ignoreDirs[d.Name()] || strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}

		ext := strings.ToLower(filepath.Ext(path))
		lang, supported := supportedExtensions[ext]
		if !supported {
			return nil
		}

		info, err := d.Info()
		if err != nil || info.Size() > maxFileSize || info.Size() == 0 {
			return nil
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}

		code := string(data)
		relPath, _ := filepath.Rel(dirPath, path)
		title := relPath

		snippet := &Snippet{
			Title:    title,
			Code:     code,
			Language: lang,
			FilePath: path,
			Tags:     autoTag(code, lang),
		}

		s.Add(snippet)
		count++
		return nil
	})

	return count, err
}

// ─── Persistence (JSON) ───────────────────────────────────────

type persistData struct {
	Snippets []*Snippet `json:"snippets"`
}

func (s *RAGStore) save() {
	if s.savePath == "" {
		return
	}

	var snippets []*Snippet
	for _, snip := range s.snippets {
		snippets = append(snippets, snip)
	}

	data, err := json.MarshalIndent(persistData{Snippets: snippets}, "", "  ")
	if err != nil {
		log.Printf("RAG save error: %v", err)
		return
	}

	os.MkdirAll(filepath.Dir(s.savePath), 0755)
	if err := os.WriteFile(s.savePath, data, 0644); err != nil {
		log.Printf("RAG save error: %v", err)
	}
}

func (s *RAGStore) load() {
	if s.savePath == "" {
		return
	}

	data, err := os.ReadFile(s.savePath)
	if err != nil {
		return // file doesn't exist yet
	}

	var pd persistData
	if err := json.Unmarshal(data, &pd); err != nil {
		log.Printf("RAG load error: %v", err)
		return
	}

	for _, snip := range pd.Snippets {
		snip.tfVector = computeTF(tokenize(snip.Code + " " + snip.Title + " " + strings.Join(snip.Tags, " ")))
		s.snippets[snip.ID] = snip
		s.indexSnippet(snip)
	}

	log.Printf("📚 RAG loaded %d snippets from %s", len(s.snippets), s.savePath)
}

// ═══════════════════════════════════════════════════════════════
// NLP Utilities
// ═══════════════════════════════════════════════════════════════

// Tokenize splits text into normalized tokens, handling code identifiers
func tokenize(text string) []string {
	// Split camelCase BEFORE lowering
	var expanded strings.Builder
	for i, r := range text {
		if i > 0 && !unicode.IsUpper(rune(text[i-1])) && unicode.IsUpper(r) {
			expanded.WriteRune(' ')
		}
		if r == '_' || r == '-' || r == '.' || r == '/' || r == '\\' {
			expanded.WriteRune(' ')
		} else {
			expanded.WriteRune(r)
		}
	}

	lowered := strings.ToLower(expanded.String())
	words := strings.Fields(lowered)
	var tokens []string

	for _, w := range words {
		// Remove non-alphanumeric edges
		w = strings.TrimFunc(w, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		})
		if len(w) < 2 {
			continue
		}
		if stopWords[w] {
			continue
		}
		tokens = append(tokens, w)
	}

	return tokens
}

func computeTF(tokens []string) map[string]float64 {
	tf := make(map[string]float64)
	for _, t := range tokens {
		tf[t]++
	}
	total := float64(len(tokens))
	if total == 0 {
		return tf
	}
	for k := range tf {
		tf[k] = tf[k] / total
	}
	return tf
}

func cosineSimilarity(a, b map[string]float64) float64 {
	var dot, normA, normB float64
	for k, v := range a {
		dot += v * b[k]
		normA += v * v
	}
	for _, v := range b {
		normB += v * v
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// ─── Language Detection ────────────────────────────────────────

func detectLanguage(code, filePath string) string {
	if filePath != "" {
		for e, lang := range supportedExtensions {
			if strings.HasSuffix(strings.ToLower(filePath), e) {
				return lang
			}
		}
	}

	// Heuristic detection from content
	codeLower := strings.ToLower(code)
	switch {
	case strings.Contains(code, "<!DOCTYPE html") || strings.Contains(code, "<html"):
		return "html"
	case strings.Contains(code, "def ") && strings.Contains(code, "import "):
		return "python"
	case strings.Contains(code, "func ") && strings.Contains(code, "package "):
		return "go"
	case strings.Contains(code, "function ") || strings.Contains(code, "const ") || strings.Contains(code, "=>"):
		return "javascript"
	case strings.Contains(codeLower, "select ") && strings.Contains(codeLower, "from "):
		return "sql"
	case strings.Contains(code, "#!/bin/bash") || strings.Contains(code, "#!/bin/sh"):
		return "bash"
	default:
		return "text"
	}
}

// ─── Auto-tagging ──────────────────────────────────────────────

func autoTag(code, language string) []string {
	tags := []string{language}
	codeLower := strings.ToLower(code)

	patterns := map[string][]string{
		"crud":       {"get", "post", "put", "delete", "create", "read", "update"},
		"api":        {"endpoint", "route", "handler", "request", "response", "/api/"},
		"auth":       {"login", "logout", "token", "jwt", "password", "auth"},
		"database":   {"select", "insert", "update", "delete from", "create table", "sqlite", "postgres", "mysql"},
		"frontend":   {"<div", "<button", "<form", "document.getelementby", "queryselector", "addeventlistener"},
		"tailwind":   {"tailwindcss", "class=\"", "className=\"", "bg-", "text-", "flex ", "grid "},
		"react":      {"usestate", "useeffect", "import react", "jsx", "component"},
		"flask":      {"flask", "@app.route", "flask_cors"},
		"express":    {"express()", "app.get", "app.post", "req, res"},
		"websocket":  {"websocket", "ws://", "wss://", "socket.io"},
		"testing":    {"test", "assert", "expect", "describe", "it("},
		"docker":     {"dockerfile", "docker-compose", "container"},
		"responsive": {"@media", "min-width", "max-width", "responsive"},
		"animation":  {"animation", "transition", "transform", "keyframes"},
		"chart":      {"chart", "graph", "d3.", "recharts", "canvas"},
		"form":       {"<form", "<input", "submit", "validation"},
	}

	for tag, keywords := range patterns {
		for _, kw := range keywords {
			if strings.Contains(codeLower, kw) {
				tags = append(tags, tag)
				break
			}
		}
	}

	// Dedupe
	seen := make(map[string]bool)
	var unique []string
	for _, t := range tags {
		if !seen[t] {
			seen[t] = true
			unique = append(unique, t)
		}
	}
	return unique
}

// ─── Stop Words ────────────────────────────────────────────────

var stopWords = map[string]bool{
	"the": true, "is": true, "at": true, "of": true, "on": true,
	"and": true, "or": true, "to": true, "in": true, "it": true,
	"an": true, "as": true, "by": true, "be": true, "if": true,
	"do": true, "no": true, "so": true, "up": true, "he": true,
	"we": true, "my": true, "me": true, "am": true, "this": true,
	"that": true, "with": true, "for": true, "from": true, "not": true,
	"but": true, "are": true, "was": true, "were": true, "been": true,
	"have": true, "has": true, "had": true, "will": true, "would": true,
	"could": true, "should": true, "may": true, "can": true, "did": true,
	"does": true, "its": true, "let": true, "put": true, "say": true,
	"she": true, "too": true, "use": true, "her": true, "all": true,
	"each": true, "which": true, "their": true, "there": true, "then": true,
	"than": true, "them": true, "these": true, "some": true, "what": true,
	"when": true, "who": true, "how": true, "any": true, "new": true,
	"var": true, "nil": true, "null": true, "true": true, "false": true,
}

// ─── Helpers ───────────────────────────────────────────────────

func topN(m map[string]int, n int) map[string]int {
	type kv struct {
		k string
		v int
	}
	var sorted []kv
	for k, v := range m {
		sorted = append(sorted, kv{k, v})
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].v > sorted[j].v })

	result := make(map[string]int)
	for i, s := range sorted {
		if i >= n {
			break
		}
		result[s.k] = s.v
	}
	return result
}

// ═══════════════════════════════════════════════════════════════
// Ollama Embedding Support (optional, uses generate endpoint)
// ═══════════════════════════════════════════════════════════════

type EmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

func getOllamaEmbedding(baseURL, text string) ([]float64, error) {
	reqBody, _ := json.Marshal(EmbeddingRequest{
		Model:  "qwen2.5-coder:0.5b",
		Prompt: text,
	})

	resp, err := httpClient.Post(baseURL+"/api/embeddings", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Embedding, nil
}
