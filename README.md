# Claude Chatbot

A Claude-powered chatbot that can dynamically modify the web page it lives in by executing JavaScript via a `run_js` tool. Instead of just answering questions, Claude can build interactive UIs, games, visualizations — anything you can do with JS — all while maintaining a persistent chat interface.

## Setup

```bash
pip install flask anthropic python-dotenv
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
```

Run:
```bash
python app.py
```

Open http://localhost:5000

## How it works

- Claude has a `run_js` tool that executes JavaScript in your browser via `eval()`
- Text responses render as HTML inside chat cells (Mathematica notebook style)
- Tool calls stream token-by-token so you can watch the code being written
- `run_js` returns the eval result back to Claude, so it can read DOM state and react to errors
- Chat history persists to disk as JSON files

## Routes

| Route | Description |
|-------|-------------|
| `/` | Opens most recent chat |
| `/new` | Starts a fresh chat |
| `/?chat=<id>` | Opens a specific chat |
| `/picker` | Chat list page with open/delete |

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send message. Body: `{"message": "...", "chat_id": "..."}`. Returns SSE stream |
| `/chats` | GET | List all saved chats |
| `/chats/<id>` | GET | Get a specific chat with full message history |
| `/chats/<id>` | DELETE | Delete a chat |
| `/reset` | POST | Get a fresh chat ID |
| `/tool_result/<id>` | POST | Browser posts eval results back here |

## Architecture

```
app.py                  Flask server, Anthropic streaming, tool loop
templates/index.html    Chat interface with SSE stream consumer
templates/picker.html   Chat list page
chats/                  Saved chat JSON files (gitignored)
```

The server streams responses via SSE. Each event is `data: {"type": "...", ...}\n\n`:

- `text_start` / `text_delta` — assistant text streaming into a chat cell
- `tool_start` / `tool_delta` — tool input JSON streaming into a code block
- `js` — final JS code to eval, with `tool_id` for result round-trip
- `tool_output` — result from server-side tool execution
- `error` — error message
- `done` — stream complete, includes `chat_id`

When a chat is loaded via `/?chat=<id>`, all previous `run_js` calls are replayed to rebuild DOM state.

## TODO

- [ ] Sandbox integration ([anoek/sandbox](https://github.com/anoek/sandbox)) for per-chat isolated filesystems
- [ ] Server-side tools (bash, read/write/edit files, grep) running inside sandboxes
- [ ] Model picker
