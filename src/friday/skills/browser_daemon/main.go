package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/launcher"
	"github.com/go-rod/rod/lib/proto"
	"github.com/go-shiori/go-readability"
)

type ManagedBrowser struct {
	Browser *rod.Browser
	Pages   map[string]*rod.Page
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
	managed := &ManagedBrowser{
		Browser: b,
		Pages:   make(map[string]*rod.Page),
	}
	m.browsers[key] = managed
	return managed, nil
}

func (m *BrowserManager) HasActiveBrowsers() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.browsers) > 0
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
	PageID   string `json:"page_id"`
}

type CloseRequest struct {
	Profile string `json:"profile"`
	PageID  string `json:"page_id"`
}

type Response struct {
	Success bool     `json:"success"`
	PageID  string   `json:"page_id,omitempty"`
	Content string   `json:"content,omitempty"`
	Message string   `json:"message,omitempty"`
	Pages   []string `json:"pages,omitempty"`
}

func main() {
	manager := NewBrowserManager()

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		// 1. If we already have managed browsers, we are definitely OK
		if manager.HasActiveBrowsers() {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("OK"))
			return
		}

		// 2. Otherwise, check if a browser can be found on the system
		// launcher.LookPath() is a lightweight check that doesn't start a process
		if _, found := launcher.LookPath(); found {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("OK"))
		} else {
			// Return 503 so friday doctor knows it's not just the server that's the issue
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte("ERROR: No compatible browser (Chrome/Chromium) found. Please install one to use browser skills."))
		}
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
		page.MustWaitLoad()

		pageID := fmt.Sprintf("p_%d", time.Now().UnixNano())
		managed.Pages[pageID] = page

		html := page.MustHTML()
		
		// Use readability to extract useful text
		parsedURL, _ := url.Parse(req.URL)
		article, err := readability.FromReader(strings.NewReader(html), parsedURL)
		var content string
		if err == nil {
			content = article.TextContent
		} else {
			content = page.MustElement("body").MustText()
		}

		json.NewEncoder(w).Encode(Response{
			Success: true,
			PageID:  pageID,
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

		managed, err := manager.GetBrowser(req.Profile, true)
		if err != nil {
			json.NewEncoder(w).Encode(Response{Success: false, Message: err.Error()})
			return
		}

		managed.Mu.Lock()
		defer managed.Mu.Unlock()

		var page *rod.Page
		if req.PageID != "" {
			page = managed.Pages[req.PageID]
		} else {
			// Fallback to first page
			pages, _ := managed.Browser.Pages()
			if len(pages) > 0 {
				page = pages[0]
			}
		}

		if page == nil {
			json.NewEncoder(w).Encode(Response{Success: false, Message: "Page not found"})
			return
		}

		switch req.Type {
		case "click":
			err = page.MustElement(req.Selector).Click(proto.InputMouseButtonLeft, 1)
		case "type":
			err = page.MustElement(req.Selector).Input(req.Value)
		case "content":
			// Just get content
		default:
			err = fmt.Errorf("unknown action type: %s", req.Type)
		}

		if err != nil {
			json.NewEncoder(w).Encode(Response{Success: false, Message: err.Error()})
		} else {
			html := page.MustHTML()
			parsedURL, _ := url.Parse(page.MustInfo().URL)
			article, err := readability.FromReader(strings.NewReader(html), parsedURL)
			content := ""
			if err == nil {
				content = article.TextContent
			}
			json.NewEncoder(w).Encode(Response{Success: true, Content: content})
		}
	})

	http.HandleFunc("/pages", func(w http.ResponseWriter, r *http.Request) {
		profile := r.URL.Query().Get("profile")
		if profile == "" {
			profile = "default"
		}

		managed, err := manager.GetBrowser(profile, true)
		if err != nil {
			json.NewEncoder(w).Encode(Response{Success: false, Message: err.Error()})
			return
		}

		managed.Mu.Lock()
		defer managed.Mu.Unlock()

		var pageIDs []string
		for id := range managed.Pages {
			pageIDs = append(pageIDs, id)
		}

		json.NewEncoder(w).Encode(Response{Success: true, Pages: pageIDs})
	})

	http.HandleFunc("/close", func(w http.ResponseWriter, r *http.Request) {
		var req CloseRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if req.Profile == "" {
			req.Profile = "default"
		}

		managed, err := manager.GetBrowser(req.Profile, true)
		if err != nil {
			json.NewEncoder(w).Encode(Response{Success: false, Message: err.Error()})
			return
		}

		managed.Mu.Lock()
		defer managed.Mu.Unlock()

		if page, ok := managed.Pages[req.PageID]; ok {
			page.Close()
			delete(managed.Pages, req.PageID)
			json.NewEncoder(w).Encode(Response{Success: true})
		} else {
			json.NewEncoder(w).Encode(Response{Success: false, Message: "Page not found"})
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
