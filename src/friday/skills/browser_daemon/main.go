package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/launcher"
	"github.com/go-shiori/go-readability"
)

type ManagedBrowser struct {
	Browser *rod.Browser
	Mu      sync.Mutex
}

type BrowserManager struct {
	browsers map[string]*ManagedBrowser
	mu       sync.Mutex
}

func NewBrowserManager() *BrowserManager {
	return &BrowserManager{
		browsers: make(map[string]*ManagedBrowser),
	}
}

func (m *BrowserManager) GetBrowser(profile string, headless bool) (*ManagedBrowser, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	key := fmt.Sprintf("%s-%v", profile, headless)
	if b, ok := m.browsers[key]; ok {
		return b, nil
	}

	l := launcher.New()
	if profile != "default" {
		l.UserDataDir(fmt.Sprintf("./profiles/%s", profile))
	}
	l.Headless(headless)

	u, err := l.Launch()
	if err != nil {
		return nil, err
	}

	b := rod.New().ControlURL(u).MustConnect()
	managed := &ManagedBrowser{Browser: b}
	m.browsers[key] = managed
	return managed, nil
}

type NavigateRequest struct {
	URL      string `json:"url"`
	Profile  string `json:"profile"`
	Headless bool   `json:"headless"`
}

type ActionRequest struct {
	Type     string `json:"type"`
	Selector string `json:"selector"`
	Value    string `json:"value"`
	Profile  string `json:"profile"`
}

type Response struct {
	Success bool   `json:"success"`
	Content string `json:"content,omitempty"`
	Message string `json:"message,omitempty"`
}

func main() {
	manager := NewBrowserManager()

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	http.HandleFunc("/navigate", func(w http.ResponseWriter, r *http.Request) {
		var req NavigateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if req.Profile == "" {
			req.Profile = "default"
		}

		managed, err := manager.GetBrowser(req.Profile, req.Headless)
		if err != nil {
			json.NewEncoder(w).Encode(Response{Success: false, Message: err.Error()})
			return
		}

		managed.Mu.Lock()
		defer managed.Mu.Unlock()

		page := managed.Browser.MustPage(req.URL)
		defer page.Close()
		page.MustWaitLoad()

		html := page.MustHTML()
		
		// Use readability to extract useful text
		article, err := readability.FromReader(strings.NewReader(html), req.URL)
		var content string
		if err == nil {
			content = article.TextContent
		} else {
			// Fallback: simple text extraction
			content = page.MustElement("body").MustText()
		}

		json.NewEncoder(w).Encode(Response{
			Success: true,
			Content: strings.TrimSpace(content),
		})
	})

	http.HandleFunc("/action", func(w http.ResponseWriter, r *http.Request) {
		var req ActionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if req.Profile == "" {
			req.Profile = "default"
		}

		// Actions usually happen on the active page. For simplicity, we assume the last opened page or search by URL.
		// In a real daemon, we might need a PageID. For now, we'll just use the last opened page of the browser.
		managed, err := manager.GetBrowser(req.Profile, true) // Action mode usually headless or uses current state
		if err != nil {
			json.NewEncoder(w).Encode(Response{Success: false, Message: err.Error()})
			return
		}

		managed.Mu.Lock()
		defer managed.Mu.Unlock()

		pages, _ := managed.Browser.Pages()
		if len(pages) == 0 {
			json.NewEncoder(w).Encode(Response{Success: false, Message: "No active pages"})
			return
		}
		page := pages[0]

		switch req.Type {
		case "click":
			err = page.MustElement(req.Selector).Click(nil, 1)
		case "type":
			err = page.MustElement(req.Selector).Input(req.Value)
		default:
			err = fmt.Errorf("unknown action type: %s", req.Type)
		}

		if err != nil {
			json.NewEncoder(w).Encode(Response{Success: false, Message: err.Error()})
		} else {
			json.NewEncoder(w).Encode(Response{Success: true})
		}
	})

	log.Println("Friday Browser Daemon starting on :9000")
	srv := &http.Server{
		Addr:         ":9000",
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
	}
	log.Fatal(srv.ListenAndServe())
}
