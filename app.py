from dotenv import load_dotenv
load_dotenv()

import json
import os
import shlex
import subprocess
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

SANDBOX_TIMEOUT = 30

SYSTEM_PROMPT = """\
You are a chatbot embedded in a web page. You have multiple ways to respond:

1. **HTML**: Your responses are rendered as raw HTML inside chat bubbles. \
Write HTML directly (e.g. <p>, <strong>, <ul>, <code>), NOT markdown. You can use this to add \
all kinds of interactive elements!

2. **run_js tool**: Execute JavaScript in the user's browser to dynamically modify the page. \
Use this to build interactive experiences, change styles, add elements, create games, \
inject canvas graphics, load CDN libraries, etc.

3. **Sandbox tools**: You have a sandboxed Linux environment with an isolated overlay filesystem. \
Use bash, read_file, write_file, edit_file, list_files, and grep to write code, run programs, \
install packages, and manage files. Each chat gets its own sandbox — changes are isolated and \
persist across messages within the same chat. The sandbox starts in /home.

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
- Sandbox tools return stdout/stderr. Use bash for anything you'd do in a terminal. \
Use read_file/write_file/edit_file for precise file operations. Use grep to search code.
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
    },
    {
        "name": "bash",
        "description": (
            "Run a shell command inside a sandboxed Linux environment. Returns stdout and stderr. "
            "The sandbox has an isolated overlay filesystem — changes persist within the chat "
            "but don't affect the host. Has network access. Use for running code, installing "
            "packages, compiling, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"}
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file from the sandbox filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the sandbox. Creates parent directories automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Find and replace a unique string in a file. old_string must appear exactly once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit"},
                "old_string": {"type": "string", "description": "Exact string to find (must be unique in file)"},
                "new_string": {"type": "string", "description": "Replacement string"}
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory (up to 3 levels deep, ignoring hidden files).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list (default: current dir)"}
            },
        },
    },
    {
        "name": "grep",
        "description": "Search for a regex pattern in files. Returns matching lines with paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "File or directory to search (default: current dir)"}
            },
            "required": ["pattern"],
        },
    },
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


def sandbox_exec(chat_id, args, input_data=None, timeout=SANDBOX_TIMEOUT):
    """Run a command inside the chat's sandbox."""
    cmd = ["sandbox", "--name", f"chat-{chat_id}", "--net=host"] + args
    try:
        result = subprocess.run(
            cmd, input=input_data, capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1


def execute_sandbox_tool(chat_id, tool_name, tool_input):
    """Execute a sandbox tool and return the result string."""
    if tool_name == "bash":
        stdout, stderr, code = sandbox_exec(
            chat_id, ["bash", "-c", tool_input["command"]], timeout=60
        )
        output = stdout
        if stderr:
            output += ("\n" if output else "") + stderr
        if code != 0 and not output:
            output = f"Exit code: {code}"
        return output or "(no output)"

    elif tool_name == "read_file":
        stdout, stderr, code = sandbox_exec(chat_id, ["cat", tool_input["path"]])
        if code != 0:
            return f"Error: {stderr.strip()}"
        return stdout

    elif tool_name == "write_file":
        path = tool_input["path"]
        content = tool_input["content"]
        safe_path = shlex.quote(path)
        stdout, stderr, code = sandbox_exec(
            chat_id,
            ["sh", "-c", f"mkdir -p \"$(dirname {safe_path})\" && cat > {safe_path}"],
            input_data=content,
        )
        if code != 0:
            return f"Error: {stderr.strip()}"
        return f"Wrote {len(content)} bytes to {path}"

    elif tool_name == "edit_file":
        path = tool_input["path"]
        old_string = tool_input["old_string"]
        new_string = tool_input["new_string"]
        stdout, stderr, code = sandbox_exec(chat_id, ["cat", path])
        if code != 0:
            return f"Error reading {path}: {stderr.strip()}"
        content = stdout
        if old_string not in content:
            return f"Error: old_string not found in {path}"
        count = content.count(old_string)
        if count > 1:
            return f"Error: old_string appears {count} times in {path}. Must be unique."
        new_content = content.replace(old_string, new_string, 1)
        safe_path = shlex.quote(path)
        stdout, stderr, code = sandbox_exec(
            chat_id, ["sh", "-c", f"cat > {safe_path}"], input_data=new_content,
        )
        if code != 0:
            return f"Error writing {path}: {stderr.strip()}"
        return f"Edited {path}"

    elif tool_name == "list_files":
        path = tool_input.get("path", ".")
        stdout, stderr, code = sandbox_exec(
            chat_id, ["find", path, "-maxdepth", "3", "-not", "-path", "*/.*", "-not", "-name", ".*"]
        )
        if code != 0 and stderr:
            return f"Error: {stderr.strip()}"
        return stdout.strip() or "(empty directory)"

    elif tool_name == "grep":
        pattern = tool_input["pattern"]
        path = tool_input.get("path", ".")
        stdout, stderr, code = sandbox_exec(
            chat_id, ["grep", "-rn", pattern, path]
        )
        if code == 1:
            return "No matches found"
        if code != 0 and stderr:
            return f"Error: {stderr.strip()}"
        return stdout.strip()

    return f"Error: Unknown tool {tool_name}"


SANDBOX_TOOL_NAMES = {t["name"] for t in TOOLS if t["name"] != "run_js"}


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

    messages = load_messages(chat_id)
    messages.append({"role": "user", "content": user_message})

    def generate():
        try:
            while True:
                current_block_type = None
                current_tool_name = None
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
                                current_tool_name = event.content_block.name
                                tool_id = event.content_block.id
                                tool_input_parts = []
                                yield sse({"type": "tool_start", "name": current_tool_name})

                        elif event.type == "content_block_delta":
                            if event.delta.type == "text_delta":
                                yield sse({"type": "text_delta", "content": event.delta.text})
                            elif event.delta.type == "input_json_delta":
                                tool_input_parts.append(event.delta.partial_json)
                                yield sse({"type": "tool_delta", "content": event.delta.partial_json})

                        elif event.type == "content_block_stop":
                            if current_block_type == "tool_use" and tool_input_parts:
                                tool_input = json.loads("".join(tool_input_parts))
                                if current_tool_name == "run_js":
                                    code = tool_input.get("code", "")
                                    if code:
                                        pending_events[tool_id] = threading.Event()
                                        yield sse({"type": "js", "code": code, "tool_id": tool_id})
                            current_block_type = None
                            current_tool_name = None

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
                    if block.type != "tool_use":
                        continue
                    if block.name == "run_js":
                        ev = pending_events.get(block.id)
                        if ev:
                            ev.wait(timeout=30)
                        result = pending_results.pop(block.id, "OK")
                        pending_events.pop(block.id, None)
                    elif block.name in SANDBOX_TOOL_NAMES:
                        result = execute_sandbox_tool(chat_id, block.name, block.input)
                        if len(result) > 10000:
                            result = result[:10000] + "\n... (truncated)"
                        yield sse({"type": "tool_output", "name": block.name, "tool_id": block.id, "result": result})
                    else:
                        result = f"Error: Unknown tool {block.name}"

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
