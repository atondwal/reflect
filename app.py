from dotenv import load_dotenv
load_dotenv()

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template, redirect, Response
import anthropic

app = Flask(__name__)
client = anthropic.Anthropic()

CHATS_DIR = os.path.join(os.path.dirname(__file__), "chats")
os.makedirs(CHATS_DIR, exist_ok=True)

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
- GET /chats → {"chats": [{"id", "title", "updated_at"}]}
- GET /chats/<id> → {"id", "title", "updated_at", "messages": [...]}
- DELETE /chats/<id> → deletes a chat
Chats auto-save after each exchange. Title is derived from the first user message. \
Navigate to a chat with /?chat=<id>. Navigate to /new for a new chat. \
/picker shows all saved chats. / reopens the most recent chat. \
The current chat ID is in the JS variable `chatId` (null if new).\
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


def save_chat(chat_id, messages):
    title = "New chat"
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], str):
            title = msg["content"][:80]
            break
    data = {
        "id": chat_id,
        "title": title,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
    }
    with open(chat_path(chat_id), "w") as f:
        json.dump(data, f)


def load_messages(chat_id):
    path = chat_path(chat_id)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("messages", [])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/new")
def new_chat_page():
    return redirect("/?new")


@app.route("/picker")
def picker():
    return render_template("picker.html")


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
    return jsonify({"chats": chats})


@app.route("/chats/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    path = chat_path(chat_id)
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    path = chat_path(chat_id)
    if os.path.exists(path):
        os.remove(path)
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    chat_id = request.json.get("chat_id")

    if not user_message:
        return Response(
            sse({"type": "error", "content": "Empty message."}) + sse({"type": "done"}),
            mimetype="text/event-stream",
        )

    if not chat_id:
        chat_id = uuid.uuid4().hex[:12]

    # Load existing history for this chat
    messages = load_messages(chat_id)
    messages.append({"role": "user", "content": user_message})

    def generate():
        try:
            while True:
                current_block_type = None
                tool_id = None
                tool_input_parts = []
                stop_reason = None

                with client.messages.stream(
                    model="claude-opus-4-6",
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
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
                                    yield sse({"type": "js", "code": code, "tool_id": tool_id})
                            current_block_type = None

                        elif event.type == "message_delta":
                            stop_reason = event.delta.stop_reason

                    final_message = stream.get_final_message()

                messages.append({
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
                    messages.append({"role": "user", "content": tool_results})
                else:
                    break

        except Exception as e:
            yield sse({"type": "error", "content": str(e)})

        save_chat(chat_id, messages)
        yield sse({"type": "done", "chat_id": chat_id})

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route("/reset", methods=["POST"])
def reset():
    return jsonify({"status": "ok", "chat_id": uuid.uuid4().hex[:12]})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
