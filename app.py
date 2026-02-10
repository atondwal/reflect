from dotenv import load_dotenv
load_dotenv()

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template, Response
import anthropic

app = Flask(__name__)
client = anthropic.Anthropic()

CHATS_DIR = os.path.join(os.path.dirname(__file__), "chats")
os.makedirs(CHATS_DIR, exist_ok=True)

# Current chat state
current_chat_id = None
conversation_history = []

# Pending tool results from the browser
pending_results = {}   # tool_id -> result string
pending_events = {}    # tool_id -> threading.Event

SYSTEM_PROMPT = """\
You are a chatbot embedded in a web page. You have two ways to respond:

1. **HTML**: Your responses are rendered as raw HTML inside chat bubbles. \
Write HTML directly (e.g. <p>, <strong>, <ul>, <code>), NOT markdown. You can use this to add \
all kinds of interactive elements!

2. **run_js tool**: Execute JavaScript in the user's browser to dynamically modify the page. \
Use this to build interactive experiences, change styles, add elements, create games, \
inject canvas graphics, load CDN libraries, etc.

Important details:
- The chat interface lives in #chat-container (messages) and #input-area (input + button). \
You can restyle these, but NEVER fully cover, hide, or obscure them. The chat must always remain \
visible, accessible, and functional. Do not place elements on top of it or set its display/visibility \
to hidden. But if you want to show something else, it's encouraged to move it out of the way as long \
as it remains acessible.
- You can load external libraries by injecting <script> tags into document.head. \
Wait for onload before using them.
- You can call run_js multiple times in one turn to build things up incrementally.
- run_js returns the result of the last expression in your code (like a browser console). \
Use this to read DOM state, check values, or verify your changes worked. If the code throws, \
you get the error message back.
- Do NOT use alert() or prompt() — they block the browser. Use the DOM or console instead.
- You have no default theme or styling opinions — you decide everything about look and feel.
- Be creative and have fun with it.

Backend API (use via fetch() in run_js):
- GET /chats → {"chats": [{"id", "title", "updated_at"}], "current": "id"}
- POST /chats/<id>/load → {"status": "ok", "id", "messages": [...]} — switches to a saved chat
- DELETE /chats/<id> → deletes a chat
- POST /reset → saves current chat, starts a new one
Chats auto-save after each exchange. Title is derived from the first user message. \
After loading a chat or resetting, reload the page (location.reload()) to sync the frontend.\
"""

TOOLS = [
    {
        "name": "run_js",
        "description": (
            "Execute JavaScript code in the user's browser. Use this to modify the page: "
            "add/remove DOM elements, change styles, inject scripts, create canvases, "
            "build interactive UIs, etc. Returns the result of the last expression "
            "(or the error message if it threw). The chat interface lives in "
            "#chat-container and #input-area — keep them visible and functional."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "JavaScript code to execute in the browser via eval()",
                }
            },
            "required": ["code"],
        },
    }
]


def sse(data):
    return f"data: {json.dumps(data)}\n\n"


def serialize_block(block):
    """Serialize a content block to only the fields the API accepts."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return block.model_dump()


def chat_path(chat_id):
    return os.path.join(CHATS_DIR, f"{chat_id}.json")


def save_chat():
    """Save current chat to disk."""
    if not current_chat_id:
        return
    # Derive title from first user message
    title = "New chat"
    for msg in conversation_history:
        if msg["role"] == "user" and isinstance(msg["content"], str):
            title = msg["content"][:80]
            break
    data = {
        "id": current_chat_id,
        "title": title,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": conversation_history,
    }
    with open(chat_path(current_chat_id), "w") as f:
        json.dump(data, f)


def load_chat(chat_id):
    """Load a chat from disk. Returns True on success."""
    global current_chat_id, conversation_history
    path = chat_path(chat_id)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        data = json.load(f)
    current_chat_id = data["id"]
    conversation_history = data["messages"]
    return True


def new_chat():
    """Start a fresh chat."""
    global current_chat_id, conversation_history
    current_chat_id = uuid.uuid4().hex[:12]
    conversation_history = []
    return current_chat_id


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/tool_result/<tool_id>", methods=["POST"])
def receive_tool_result(tool_id):
    data = request.json
    pending_results[tool_id] = data.get("result", "OK")
    ev = pending_events.get(tool_id)
    if ev:
        ev.set()
    return jsonify({"status": "ok"})


@app.route("/chats", methods=["GET"])
def list_chats():
    chats = []
    for fname in os.listdir(CHATS_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(CHATS_DIR, fname)) as f:
            data = json.load(f)
        chats.append({
            "id": data["id"],
            "title": data.get("title", "Untitled"),
            "updated_at": data.get("updated_at", ""),
        })
    chats.sort(key=lambda c: c["updated_at"], reverse=True)
    return jsonify({"chats": chats, "current": current_chat_id})


@app.route("/chats/<chat_id>/load", methods=["POST"])
def load_chat_endpoint(chat_id):
    # Save current chat first
    save_chat()
    if load_chat(chat_id):
        return jsonify({"status": "ok", "id": current_chat_id, "messages": conversation_history})
    return jsonify({"status": "error", "message": "Chat not found"}), 404


@app.route("/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    global current_chat_id, conversation_history
    path = chat_path(chat_id)
    if os.path.exists(path):
        os.remove(path)
    if current_chat_id == chat_id:
        new_chat()
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    global current_chat_id
    user_message = request.json.get("message", "")
    if not user_message:
        return Response(
            sse({"type": "error", "content": "Empty message."}) + sse({"type": "done"}),
            mimetype="text/event-stream",
        )

    # Auto-create a chat if none active
    if not current_chat_id:
        new_chat()

    conversation_history.append({"role": "user", "content": user_message})

    def generate():
        try:
            while True:
                current_block_type = None
                tool_id = None
                tool_input_parts = []
                stop_reason = None
                round_tool_ids = []

                with client.messages.stream(
                    model="claude-opus-4-6",
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=conversation_history,
                ) as stream:
                    for event in stream:
                        if event.type == "content_block_start":
                            if event.content_block.type == "text":
                                current_block_type = "text"
                                yield sse({"type": "text_start"})
                            elif event.content_block.type == "tool_use":
                                current_block_type = "tool_use"
                                tool_id = event.content_block.id
                                tool_input_parts = []
                                yield sse({"type": "tool_start"})

                        elif event.type == "content_block_delta":
                            if event.delta.type == "text_delta":
                                yield sse({"type": "text_delta", "content": event.delta.text})
                            elif event.delta.type == "input_json_delta":
                                tool_input_parts.append(event.delta.partial_json)
                                yield sse({"type": "tool_delta", "content": event.delta.partial_json})

                        elif event.type == "content_block_stop":
                            if current_block_type == "tool_use" and tool_input_parts:
                                tool_input = json.loads("".join(tool_input_parts))
                                code = tool_input.get("code", "")
                                if code:
                                    pending_events[tool_id] = threading.Event()
                                    round_tool_ids.append(tool_id)
                                    yield sse({"type": "js", "code": code, "tool_id": tool_id})
                            current_block_type = None

                        elif event.type == "message_delta":
                            stop_reason = event.delta.stop_reason

                    final_message = stream.get_final_message()

                conversation_history.append({
                    "role": "assistant",
                    "content": [serialize_block(b) for b in final_message.content],
                })

                if stop_reason == "end_turn":
                    break

                tool_results = []
                for block in final_message.content:
                    if block.type == "tool_use" and block.name == "run_js":
                        ev = pending_events.get(block.id)
                        if ev:
                            ev.wait(timeout=30)
                        result = pending_results.pop(block.id, "OK")
                        pending_events.pop(block.id, None)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                if tool_results:
                    conversation_history.append({"role": "user", "content": tool_results})
                else:
                    break

        except Exception as e:
            yield sse({"type": "error", "content": str(e)})

        save_chat()
        yield sse({"type": "done"})

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route("/reset", methods=["POST"])
def reset():
    save_chat()
    chat_id = new_chat()
    return jsonify({"status": "ok", "id": chat_id})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
