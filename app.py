from dotenv import load_dotenv
load_dotenv()

import json
from flask import Flask, request, jsonify, render_template, Response
import anthropic

app = Flask(__name__)
client = anthropic.Anthropic()

conversation_history = []

SYSTEM_PROMPT = """\
You are a chatbot embedded in a web page. You have two ways to respond:

1. **Normal text**: Your text responses are rendered as raw HTML inside chat bubbles. \
Write HTML directly (e.g. <p>, <strong>, <ul>, <code>), NOT markdown. Keep responses concise.

2. **run_js tool**: Execute JavaScript in the user's browser to dynamically modify the page. \
Use this to build interactive experiences, change styles, add elements, create games, \
inject canvas graphics, load CDN libraries, etc.

Important details:
- The chat interface lives in #chat-container (messages) and #input-area (input + button). \
You can restyle these, but NEVER cover, hide, or obscure them. The chat must always remain \
visible, accessible, and functional. Do not place elements on top of it or set its display/visibility \
to hidden.
- You can load external libraries by injecting <script> tags into document.head. \
Wait for onload before using them.
- You can call run_js multiple times in one turn to build things up incrementally.
- For simple questions, just respond with text. For building/modifying things, use run_js.
- You have no default theme or styling opinions — you decide everything about look and feel.
- Be creative and have fun with it.\
"""

TOOLS = [
    {
        "name": "run_js",
        "description": (
            "Execute JavaScript code in the user's browser. Use this to modify the page: "
            "add/remove DOM elements, change styles, inject scripts, create canvases, "
            "build interactive UIs, etc. The chat interface lives in #chat-container and "
            "#input-area — you can move, restyle, or resize these but keep chat functional."
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return Response(
            sse({"type": "error", "content": "Empty message."}) + sse({"type": "done"}),
            mimetype="text/event-stream",
        )

    conversation_history.append({"role": "user", "content": user_message})

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
                                    yield sse({"type": "js", "code": code})
                            current_block_type = None

                        elif event.type == "message_delta":
                            stop_reason = event.delta.stop_reason

                    final_message = stream.get_final_message()

                # Add assistant turn to history
                conversation_history.append({
                    "role": "assistant",
                    "content": [serialize_block(b) for b in final_message.content],
                })

                if stop_reason == "end_turn":
                    break

                # Feed tool results back for next round
                tool_results = []
                for block in final_message.content:
                    if block.type == "tool_use" and block.name == "run_js":
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "OK",
                        })

                if tool_results:
                    conversation_history.append({"role": "user", "content": tool_results})
                else:
                    break

        except Exception as e:
            yield sse({"type": "error", "content": str(e)})

        yield sse({"type": "done"})

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
