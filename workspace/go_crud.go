package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
)

// Item represents a generic CRUD resource
type Item struct {
	ID          int    `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Price       float64 `json:"price"`
	Category    string `json:"category"`
	CreatedAt   string `json:"created_at"`
}

// Store provides thread-safe in-memory storage
type Store struct {
	mu     sync.RWMutex
	items  []Item
	nextID int
}

func NewStore() *Store {
	return &Store{nextID: 1}
}

func (s *Store) GetAll() []Item {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]Item, len(s.items))
	copy(result, s.items)
	return result
}

func (s *Store) GetByID(id int) (Item, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, item := range s.items {
		if item.ID == id {
			return item, true
		}
	}
	return Item{}, false
}

func (s *Store) Create(item Item) Item {
	s.mu.Lock()
	defer s.mu.Unlock()
	item.ID = s.nextID
	s.nextID++
	s.items = append(s.items, item)
	return item
}

func (s *Store) Update(id int, updated Item) (Item, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, item := range s.items {
		if item.ID == id {
			updated.ID = id
			s.items[i] = updated
			return updated, true
		}
	}
	return Item{}, false
}

func (s *Store) Delete(id int) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, item := range s.items {
		if item.ID == id {
			s.items = append(s.items[:i], s.items[i+1:]...)
			return true
		}
	}
	return false
}

// CORSMiddleware adds CORS headers
func CORSMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

// JSON helper
func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// extractID parses numeric ID from URL path like /api/items/123
func extractID(path, prefix string) (int, error) {
	idStr := strings.TrimPrefix(path, prefix)
	return strconv.Atoi(idStr)
}

func main() {
	store := NewStore()

	// Seed data
	store.Create(Item{Name: "Widget", Description: "A useful widget", Price: 9.99, Category: "tools"})
	store.Create(Item{Name: "Gadget", Description: "A fancy gadget", Price: 19.99, Category: "electronics"})

	mux := http.NewServeMux()

	mux.HandleFunc("/api/items", CORSMiddleware(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "GET":
			writeJSON(w, http.StatusOK, store.GetAll())
		case "POST":
			var item Item
			if err := json.NewDecoder(r.Body).Decode(&item); err != nil {
				writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
				return
			}
			created := store.Create(item)
			writeJSON(w, http.StatusCreated, created)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}))

	mux.HandleFunc("/api/items/", CORSMiddleware(func(w http.ResponseWriter, r *http.Request) {
		id, err := extractID(r.URL.Path, "/api/items/")
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid ID"})
			return
		}

		switch r.Method {
		case "GET":
			item, ok := store.GetByID(id)
			if !ok {
				writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
				return
			}
			writeJSON(w, http.StatusOK, item)
		case "PUT":
			var item Item
			if err := json.NewDecoder(r.Body).Decode(&item); err != nil {
				writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
				return
			}
			updated, ok := store.Update(id, item)
			if !ok {
				writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
				return
			}
			writeJSON(w, http.StatusOK, updated)
		case "DELETE":
			if !store.Delete(id) {
				writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
				return
			}
			writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}))

	fmt.Println("Server running on :8081")
	http.ListenAndServe(":8081", mux)
}
