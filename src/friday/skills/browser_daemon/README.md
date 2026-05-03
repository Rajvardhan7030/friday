# Friday Browser Daemon

This is a self-contained browser control server written in Go. It allows Friday to control a web browser (Chrome/Chromium) to navigate sites, perform actions, and extract text efficiently.

## Prerequisites

1.  **Go 1.21+**: [Download Go](https://go.dev/dl/)
2.  **Chrome/Chromium**: Ensure a browser is installed on your system.

## Setup

1.  Navigate to this directory:
    ```bash
    cd src/friday/skills/browser_daemon
    ```
2.  Install dependencies:
    ```bash
    go mod download
    ```
3.  Build the daemon:
    ```bash
    go build -o friday-browser-daemon
    ```

## Usage

Start the daemon in the background:
```bash
./friday-browser-daemon &
```
The daemon will listen on `http://localhost:9000` by default.

## API

- `POST /navigate`: `{"url": "...", "profile": "default", "headless": true}`
- `POST /action`: `{"type": "click|type", "selector": "...", "value": "...", "profile": "default"}`
- `GET /health`: Returns `OK`.
