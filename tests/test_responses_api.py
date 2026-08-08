"""End-to-end tests for the Responses API inbound adapter and upstream protocols."""

import json
import os
import sys
import threading
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# Prevent the module-level health check from touching real endpoints during tests.
_original_thread = threading.Thread


class _NoStartThread(_original_thread):
    def start(self):
        return None


threading.Thread = _NoStartThread
import api_pool_server as m
threading.Thread = _original_thread


RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS  {name}")
    except Exception as exc:
        RESULTS.append((name, False, repr(exc)))
        print(f"FAIL  {name}: {exc}")
        traceback.print_exc()


def start_server(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def request(base, method, path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()


def set_pool(base_url, protocol="openai", name="test_mock"):
    m.pool = m.APIPool(default_payload={"temperature": 0.7})
    m.pool.add_endpoint({
        "name": name,
        "base_url": base_url,
        "api_key": "sk-test",
        "model": "mock-model",
        "priority": 1,
        "timeout": 10,
        "max_retries": 0,
        "enabled": True,
        "use_proxy": False,
        "protocol": protocol,
        "is_vision": True
    })


class TextChatMock(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.calls.append(body)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}]},
                {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]},
                {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "choices": [],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
            ]
            for chunk in chunks:
                self.wfile.write(b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            resp = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": "mock-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
            }
            data = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


class ReasoningCascadeMock(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.calls.append(body)
        effort = body.get("reasoning_effort")
        if effort in ("xhigh", "high", "medium"):
            err = json.dumps({
                "error": {
                    "message": (
                        "Error from provider (Console Go): Upstream request failed: "
                        "[invalid_parameter_error] <400> InternalError.Algo.InvalidParameter: "
                        f"Range of reasoning effort, input: '{effort}', allowed: low/medium"
                    )
                }
            }).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return
        resp = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
        }
        data = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class SingleToolCallMock(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.calls.append(body)
        if body.get("parallel_tool_calls") is not False:
            err = json.dumps({
                "error": {"message": "This model only supports single tool-calls at once!"}
            }).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return
        resp = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        }
        data = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ContextOverflowMock(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.calls.append(body)
        if len(self.calls) == 1:
            err = json.dumps({
                "error": {
                    "message": "This model's maximum context length is 1048576 tokens. However, "
                               "your messages resulted in 2000000 tokens. Please reduce the length."
                }
            }).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return
        resp = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }
        data = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ToolChatMock(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.calls.append(body)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "checking"}, "finish_reason": None}]},
                {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"tool_calls": [{
                     "index": 0, "id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"Be"}}]}, "finish_reason": None}]},
                {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"tool_calls": [{
                     "index": 0, "function": {"arguments": "ijing\"}"}}]}, "finish_reason": None}]},
                {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "choices": [],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
            ]
            for chunk in chunks:
                self.wfile.write(b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            msg = {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"}
                }]
            }
            resp = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": "mock-model",
                "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
            }
            data = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


class AnthropicMock(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.calls.append(body)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 0}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "checking"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "content_block_start", "index": 1, "content_block": {
                    "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}},
                {"type": "content_block_delta", "index": 1,
                 "delta": {"type": "input_json_delta", "partial_json": "{\"city\":\"Beijing\"}"}},
                {"type": "content_block_stop", "index": 1},
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 8}},
                {"type": "message_stop"}
            ]
            for event in events:
                self.wfile.write(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            resp = {
                "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Beijing"}}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 8, "cache_read_input_tokens": 2}
            }
            data = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


class ResponsesUpstreamMock(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.calls.append(body)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            response = {
                "id": "resp_up_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "output": [{
                    "id": "fc_up_1",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_up_1",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Beijing\"}",
                    "output": None
                }],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                          "input_tokens_details": {"cached_tokens": 1}}
            }
            events = [
                ("response.created", {"type": "response.created", "response": response}),
                ("response.output_text.delta", {"type": "response.output_text.delta",
                                                "item_id": "msg_up_1", "output_index": 0,
                                                "content_index": 0, "delta": "checking"}),
                ("response.output_item.added", {"type": "response.output_item.added", "output_index": 1,
                                                "item": response["output"][0]}),
                ("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta",
                                                            "item_id": "fc_up_1", "output_index": 1,
                                                            "delta": "{\"city\":\"Beijing\"}"}),
                ("response.output_item.done", {"type": "response.output_item.done", "output_index": 1,
                                               "item": response["output"][0]}),
                ("response.completed", {"type": "response.completed", "response": response})
            ]
            for event_type, data in events:
                self.wfile.write(
                    f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
                )
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            resp = {
                "id": "resp_up_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "output": [
                    {"id": "msg_up_1", "type": "message", "status": "completed", "role": "assistant",
                     "content": [{"type": "output_text", "text": "checking", "annotations": []}]},
                    {"id": "fc_up_1", "type": "function_call", "status": "completed",
                     "call_id": "call_up_1", "name": "get_weather",
                     "arguments": "{\"city\":\"Beijing\"}", "output": None}
                ],
                "usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19,
                          "input_tokens_details": {"cached_tokens": 2}}
            }
            data = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


class ResponsesTextUpstreamMock(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.calls.append(body)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            response = {
                "id": "resp_up_text_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "output": [{
                    "id": "msg_up_text_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "reasoning_text": "thinking step",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}]
                }],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                          "input_tokens_details": {"cached_tokens": 1}}
            }
            events = [
                ("response.created", {"type": "response.created", "response": response}),
                ("response.output_text.delta", {"type": "response.output_text.delta",
                                                "item_id": "msg_up_text_1", "output_index": 0,
                                                "content_index": 0, "delta": "hel"}),
                ("response.output_text.delta", {"type": "response.output_text.delta",
                                                "item_id": "msg_up_text_1", "output_index": 0,
                                                "content_index": 0, "delta": "lo"}),
                ("response.output_item.done", {"type": "response.output_item.done", "output_index": 0,
                                               "item": response["output"][0]}),
                ("response.completed", {"type": "response.completed", "response": response})
            ]
            for event_type, data in events:
                self.wfile.write(
                    f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
                )
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            data = json.dumps({
                "id": "resp_up_text_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "output": [{
                    "id": "msg_up_text_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "reasoning_text": "thinking step",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}]
                }],
                "usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19,
                          "input_tokens_details": {"cached_tokens": 2}}
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


class ResponsesNoUsageTextMock(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.calls.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        response = {
            "id": "resp_up_nousage_1",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "output": [{
                "id": "msg_up_nousage_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}]
            }]
        }
        events = [
            ("response.created", {"type": "response.created", "response": response}),
            ("response.output_text.delta", {"type": "response.output_text.delta",
                                            "item_id": "msg_up_nousage_1", "output_index": 0,
                                            "content_index": 0, "delta": "hi"}),
            ("response.completed", {"type": "response.completed", "response": response})
        ]
        for event_type, data in events:
            self.wfile.write(
                f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
            )
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def parse_sse(raw):
    events = []
    for block in raw.decode("utf-8").split("\n\n"):
        event_type = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data = line[6:].strip()
        if event_type and data and data != "[DONE]":
            events.append((event_type, json.loads(data)))
    return events


def test_inbound_openai_text():
    server, port = start_server(TextChatMock)
    set_pool(f"http://127.0.0.1:{port}")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    payload = {
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        ],
        "instructions": "be brief",
        "reasoning": {"effort": "low"}
    }
    status, ctype, body = request(base, "POST", "/v1/responses", payload)
    assert status == 200 and ctype.startswith("application/json")
    resp = json.loads(body)
    assert resp["object"] == "response" and resp["output"][0]["content"][0]["text"] == "hello"
    assert resp["usage"]["total_tokens"] == 19
    assert resp["reasoning"]["effort"] == "low"
    sent = TextChatMock.calls[-1]
    assert sent["messages"][0]["role"] == "system" and sent["messages"][0]["content"] == "be brief"
    assert sent["messages"][1]["content"] == "hi"
    assert sent["messages"][2]["content"][0]["type"] == "image_url"
    assert sent.get("reasoning_effort") == "low" and "store" not in sent

    status, _, body = request(base, "POST", "/v1/responses", {
        "input": "json please",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "weather",
                "schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                "strict": True
            }
        }
    })
    assert status == 200
    sent = TextChatMock.calls[-1]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["name"] == "weather"

    status, ctype, body = request(base, "POST", "/v1/responses", {"input": "stream me", "stream": True})
    assert status == 200 and ctype.startswith("text/event-stream")
    events = parse_sse(body)
    event_names = [e for e, _ in events]
    assert "response.output_text.delta" in event_names and "response.completed" in event_names
    text = "".join(d["delta"] for e, d in events if e == "response.output_text.delta")
    assert text == "Hello"
    completed = next(d for e, d in events if e == "response.completed")
    assert completed["response"]["usage"]["total_tokens"] == 15
    assert b"data: [DONE]" in body
    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_reasoning_xhigh_normalized_for_upstream():
    server, port = start_server(TextChatMock)
    set_pool(f"http://127.0.0.1:{port}")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    assert m._normalize_reasoning_effort("xhigh") == "high"
    assert m._normalize_reasoning_effort("xlow") == "low"
    assert m._normalize_reasoning_effort("minimal") == "low"
    assert m._normalize_reasoning_effort("high") == "high"

    TextChatMock.calls.clear()
    status, _, body = request(base, "POST", "/v1/responses", {
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "reasoning": {"effort": "xhigh"}
    })
    assert status == 200
    resp = json.loads(body)
    assert resp["reasoning"]["effort"] == "xhigh"
    sent = TextChatMock.calls[-1]
    assert sent.get("reasoning_effort") == "high"

    TextChatMock.calls.clear()
    status, _, body = request(base, "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "xhigh"
    })
    assert status == 200
    sent = TextChatMock.calls[-1]
    assert sent.get("reasoning_effort") == "high"

    ep = m.pool.list_endpoints()[0]
    m.pool.update_endpoint(ep["id"], {"supports_xhigh": True})
    TextChatMock.calls.clear()
    status, _, body = request(base, "POST", "/v1/responses", {
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "reasoning": {"effort": "xhigh"}
    })
    assert status == 200
    sent = TextChatMock.calls[-1]
    assert sent.get("reasoning_effort") == "xhigh"

    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_reasoning_cascade_downgrade():
    ReasoningCascadeMock.calls.clear()
    server, port = start_server(ReasoningCascadeMock)
    set_pool(f"http://127.0.0.1:{port}")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, _, body = request(base, "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "xhigh"
    })
    assert status == 200
    efforts = [c.get("reasoning_effort") for c in ReasoningCascadeMock.calls]
    assert efforts == ["high", "medium", "low"]

    assert m._downgrade_reasoning_effort("max") == "high"
    assert m._downgrade_reasoning_effort("minimal") == "low"
    assert m._downgrade_reasoning_effort("none") is None
    assert m._downgrade_reasoning_effort("low") is None
    assert m._clamp_reasoning_effort("xhigh", "medium") == "medium"
    assert m._clamp_reasoning_effort("low", "medium") == "low"

    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_max_reasoning_effort_clamp():
    TextChatMock.calls.clear()
    server, port = start_server(TextChatMock)
    set_pool(f"http://127.0.0.1:{port}")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    ep = m.pool.list_endpoints()[0]
    m.pool.update_endpoint(ep["id"], {"max_reasoning_effort": "medium"})
    status, _, body = request(base, "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "xhigh"
    })
    assert status == 200
    sent = TextChatMock.calls[-1]
    assert sent.get("reasoning_effort") == "medium"

    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_single_tool_call_auto_serialize():
    SingleToolCallMock.calls.clear()
    server, port = start_server(SingleToolCallMock)
    set_pool(f"http://127.0.0.1:{port}")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, _, body = request(base, "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}],
        "parallel_tool_calls": True
    })
    assert status == 200
    assert SingleToolCallMock.calls[-1].get("parallel_tool_calls") is False

    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_empty_assistant_message_cleaned():
    TextChatMock.calls.clear()
    server, port = start_server(TextChatMock)
    set_pool(f"http://127.0.0.1:{port}")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, _, body = request(base, "POST", "/v1/chat/completions", {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "again"}
        ]
    })
    assert status == 200
    roles = [m["role"] for m in TextChatMock.calls[-1]["messages"]]
    assert roles == ["user", "user"]

    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_context_overflow_trimmed_and_retried():
    ContextOverflowMock.calls.clear()
    server, port = start_server(ContextOverflowMock)
    set_pool(f"http://127.0.0.1:{port}")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, _, body = request(base, "POST", "/v1/chat/completions", {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old-1"},
            {"role": "assistant", "content": "old-2"},
            {"role": "user", "content": "new"}
        ]
    })
    assert status == 200
    assert len(ContextOverflowMock.calls) == 2
    trimmed = ContextOverflowMock.calls[-1]["messages"]
    assert [m["role"] for m in trimmed] == ["system", "user"]
    assert trimmed[-1]["content"] == "new"
    resp = json.loads(body)
    assert "Context was trimmed" in resp["_api_pool_notices"][0]

    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_trim_context_payload_helpers():
    payload = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "user", "content": "c"}
    ]}
    p = m._trim_context_payload(payload)
    assert [x["content"] for x in p["messages"]] == ["sys", "c"]

    huge = "x" * (m.CONTEXT_TRIM_CHARS + 100)
    p = m._trim_context_payload({"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": huge}
    ]})
    assert len(p["messages"][-1]["content"]) <= m.CONTEXT_TRIM_CHARS + 20
    assert p["messages"][-1]["content"].endswith("x" * 100)

    p = m._trim_context_payload({"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [
            {"type": "text", "text": "x" * (m.CONTEXT_TRIM_CHARS + 50)},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ]}
    ]})
    parts = p["messages"][-1]["content"]
    assert not any(x.get("type") == "image_url" for x in parts)
    text_part = next(x for x in parts if x.get("type") == "text")
    assert len(text_part["text"]) <= m.CONTEXT_TRIM_CHARS + 20

    assert m._trim_context_payload({"messages": [
        {"role": "user", "content": "short"}
    ]}) is None


def test_deepseek_reasoning_echo():
    assert m._extract_reasoning_text({"reasoning_text": "step 1"}) == "step 1"
    assert m._extract_reasoning_text({"reasoning_content": "step 2"}) == "step 2"
    assert m._extract_reasoning_text({
        "content": [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "step 3"}]}]
    }) == "step 3"
    text, tool_calls, rt = m._responses_output_to_chat_message([{
        "type": "message", "role": "assistant", "reasoning_text": "deep think",
        "content": [{"type": "output_text", "text": "hello", "annotations": []}]
    }])
    assert text == "hello" and tool_calls == [] and rt == "deep think"

    ResponsesTextUpstreamMock.calls.clear()
    server, port = start_server(ResponsesTextUpstreamMock)
    set_pool(f"http://127.0.0.1:{port}", protocol="responses", name="test_resp_up")
    ep = m.pool.list_endpoints()[0]
    m.pool.update_endpoint(ep["id"], {"model": "deepseek-v4-flash"})
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, _, _ = request(base, "POST", "/v1/responses", {
        "input": [
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "prev", "annotations": []}],
             "reasoning_text": "prev think"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]
    })
    assert status == 200
    sent_input = ResponsesTextUpstreamMock.calls[-1]["input"]
    assistant_item = next(i for i in sent_input
                          if i.get("type") == "message" and i.get("role") == "assistant")
    assert assistant_item.get("reasoning_text") == "prev think"

    ResponsesTextUpstreamMock.calls.clear()
    status, _, _ = request(base, "POST", "/v1/responses", {
        "input": [
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "prev", "annotations": []}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]
    })
    assert status == 200
    sent_input = ResponsesTextUpstreamMock.calls[-1]["input"]
    assistant_item = next(i for i in sent_input
                          if i.get("type") == "message" and i.get("role") == "assistant")
    assert assistant_item.get("reasoning_text") == ""

    chat_server, chat_port = start_server(TextChatMock)
    TextChatMock.calls.clear()
    set_pool(f"http://127.0.0.1:{chat_port}")
    ep = m.pool.list_endpoints()[0]
    m.pool.update_endpoint(ep["id"], {"model": "deepseek-v4-flash"})
    status, _, _ = request(base, "POST", "/v1/chat/completions", {
        "messages": [
            {"role": "assistant", "content": "prev"},
            {"role": "user", "content": "hi"}
        ]
    })
    assert status == 200
    assistant = next(mmsg for mmsg in TextChatMock.calls[-1]["messages"] if mmsg["role"] == "assistant")
    assert assistant.get("reasoning_content") == ""

    chat_server.shutdown()
    chat_server.server_close()
    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_openai_tool_calls():
    ToolChatMock.calls.clear()
    server, port = start_server(ToolChatMock)
    set_pool(f"http://127.0.0.1:{port}")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, _, body = request(base, "POST", "/v1/responses", {"input": "weather?"})
    resp = json.loads(body)
    fc = resp["output"][1]
    assert fc["type"] == "function_call" and fc["name"] == "get_weather"
    assert json.loads(fc["arguments"])["city"] == "Beijing"
    assert resp["usage"]["total_tokens"] == 19

    status, _, body = request(base, "POST", "/v1/responses", {"input": "weather?", "stream": True})
    events = parse_sse(body)
    names = [e for e, _ in events]
    assert "response.function_call_arguments.delta" in names
    assert "response.function_call_arguments.done" in names
    done = next(d for e, d in events if e == "response.function_call_arguments.done")
    assert json.loads(done["arguments"])["city"] == "Beijing"

    status, _, body = request(base, "POST", "/v1/chat/completions",
                              {"messages": [{"role": "user", "content": "weather?"}]})
    chat = json.loads(body)
    assert chat["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert chat["usage"]["total_tokens"] == 19

    status, _, body = request(base, "POST", "/v1/responses", {"input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather"}]},
        {"type": "function_call", "call_id": "call_prev", "name": "get_weather",
         "arguments": "{\"city\":\"Beijing\"}"},
        {"type": "function_call_output", "call_id": "call_prev", "output": "sunny"}
    ]})
    assert status == 200
    sent = ToolChatMock.calls[-1]
    assistant = next(m for m in sent["messages"] if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "call_prev"
    tool_msg = next(m for m in sent["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_prev" and tool_msg["content"] == "sunny"
    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_anthropic_tool_calls():
    AnthropicMock.calls.clear()
    server, port = start_server(AnthropicMock)
    set_pool(f"http://127.0.0.1:{port}", protocol="anthropic", name="test_claude")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    request_body = {
        "input": "weather?",
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "description": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }],
        "tool_choice": {"type": "function", "name": "get_weather"}
    }
    status, _, body = request(base, "POST", "/v1/responses", request_body)
    resp = json.loads(body)
    fc = resp["output"][1]
    assert fc["type"] == "function_call" and fc["name"] == "get_weather"
    assert resp["usage"]["total_tokens"] == 20
    sent = AnthropicMock.calls[-1]
    assert sent["tools"][0]["name"] == "get_weather"
    assert sent["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"
    assert sent["tool_choice"] == {"type": "tool", "name": "get_weather"}

    status, _, body = request(base, "POST", "/v1/responses", {"input": "weather?", "stream": True})
    events = parse_sse(body)
    fc_added = [d for e, d in events if e == "response.output_item.added"
                and d.get("item", {}).get("type") == "function_call"]
    assert len(fc_added) == 1
    done = next(d for e, d in events if e == "response.function_call_arguments.done")
    assert done["name"] == "get_weather" and json.loads(done["arguments"])["city"] == "Beijing"

    conversation = [
        {"role": "user", "content": "what weather"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"}
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "user", "content": "thanks"}
    ]
    m.pool.chat(conversation, extra_payload={})
    sent = AnthropicMock.calls[-1]
    anthropic_messages = sent["messages"]
    assert anthropic_messages[1]["content"][0]["type"] == "tool_use"
    assert anthropic_messages[1]["content"][0]["input"] == {"city": "Beijing"}
    assert anthropic_messages[2]["content"][0]["type"] == "tool_result"
    assert anthropic_messages[2]["content"][0]["content"] == "sunny"
    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_native_responses_upstream():
    server, port = start_server(ResponsesUpstreamMock)
    set_pool(f"http://127.0.0.1:{port}", protocol="responses", name="test_resp_up")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, _, body = request(base, "POST", "/v1/responses", {"input": "hi"})
    resp = json.loads(body)
    assert resp["output"][0]["content"][0]["text"] == "checking"
    assert resp["output"][1]["type"] == "function_call"
    assert resp["usage"]["total_tokens"] == 19
    assert ResponsesUpstreamMock.calls[-1]["input"][0]["type"] == "message"

    status, _, body = request(base, "POST", "/v1/responses", {"input": "hi", "stream": True})
    events = parse_sse(body)
    assert "response.function_call_arguments.done" in [e for e, _ in events]
    completed = next(d for e, d in events if e == "response.completed")
    assert completed["response"]["usage"]["total_tokens"] == 15

    status, _, body = request(base, "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "json please"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "weather",
                "schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }
    })
    assert status == 200
    sent = ResponsesUpstreamMock.calls[-1]
    assert sent["text"]["format"]["type"] == "json_schema"
    assert sent["text"]["format"]["name"] == "weather"

    status, _, body = request(base, "POST", "/v1/chat/completions",
                              {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert status == 200 and b"checking" in body
    assert b'"finish_reason": "tool_calls"' in body
    assert b"data: [DONE]" in body
    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_responses_upstream_chat_stream_finish_reason():
    ResponsesTextUpstreamMock.calls.clear()
    server, port = start_server(ResponsesTextUpstreamMock)
    set_pool(f"http://127.0.0.1:{port}", protocol="responses", name="test_resp_text_up")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, _, body = request(base, "POST", "/v1/chat/completions",
                              {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert status == 200
    assert b'"content": "hel"' in body and b'"content": "lo"' in body
    assert b'"finish_reason": "stop"' in body
    assert b"data: [DONE]" in body
    assert body.rindex(b'"finish_reason": "stop"') < body.rindex(b"data: [DONE]")
    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_responses_upstream_chat_stream_finish_without_usage():
    ResponsesNoUsageTextMock.calls.clear()
    server, port = start_server(ResponsesNoUsageTextMock)
    set_pool(f"http://127.0.0.1:{port}", protocol="responses", name="test_resp_nousage_up")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, _, body = request(base, "POST", "/v1/chat/completions",
                              {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert status == 200
    assert body.count(b'"finish_reason": "stop"') == 1
    assert body.count(b"data: [DONE]") == 1
    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_store_previous_reasoning_retrieve_delete():
    TextChatMock.calls.clear()
    server, port = start_server(TextChatMock)
    set_pool(f"http://127.0.0.1:{port}")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, _, body = request(base, "POST", "/v1/responses",
                              {"input": "first", "store": True, "reasoning": {"effort": "low"}})
    assert status == 200
    rid1 = json.loads(body)["id"]
    assert len(TextChatMock.calls[-1]["messages"]) == 1

    status, _, body = request(base, "GET", f"/v1/responses/{rid1}")
    assert status == 200 and json.loads(body)["id"] == rid1

    status, _, body = request(base, "POST", "/v1/responses",
                              {"input": "second", "previous_response_id": rid1, "store": True})
    assert status == 200
    rid2 = json.loads(body)["id"]
    assert len(TextChatMock.calls[-1]["messages"]) == 2

    status, _, body = request(base, "POST", "/v1/responses",
                              {"input": "third", "previous_response_id": rid2, "store": True, "stream": True})
    assert status == 200
    stream_id = None
    for event_type, data in parse_sse(body):
        if event_type == "response.completed":
            stream_id = data["response"]["id"]
    assert stream_id and len(TextChatMock.calls[-1]["messages"]) == 3

    status, _, body = request(base, "POST", "/v1/responses",
                              {"input": "fourth", "previous_response_id": stream_id, "store": True})
    assert status == 200 and len(TextChatMock.calls[-1]["messages"]) == 4

    status, _, body = request(base, "DELETE", f"/v1/responses/{rid1}")
    assert status == 200 and json.loads(body)["deleted"] is True
    status, _, _ = request(base, "GET", f"/v1/responses/{rid1}")
    assert status == 404
    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_error_cases():
    server, port = start_server(TextChatMock)
    set_pool(f"http://127.0.0.1:{port}")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, _, body = request(base, "POST", "/v1/responses", {})
    assert status == 400 and "input is required" in json.loads(body)["error"]["message"]
    status, _, body = request(base, "POST", "/v1/responses",
                              {"input": "x", "previous_response_id": "resp_missing"})
    assert status == 400 and "not found" in json.loads(body)["error"]["message"]
    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def test_regression_chat_and_health_call_sites():
    TextChatMock.calls.clear()
    server, port = start_server(TextChatMock)
    set_pool(f"http://127.0.0.1:{port}")
    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, _, body = request(base, "POST", "/v1/chat/completions",
                              {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and json.loads(body)["choices"][0]["message"]["content"] == "hello"

    status, _, body = request(base, "POST", "/v1/chat/completions",
                              {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert status == 200 and b"data: [DONE]" in body

    ep = m.pool.list_endpoints()[0]
    ep_obj = next(e for e in m.pool._endpoints if e.id == ep["id"])
    health = m.pool._check_one_health(ep_obj)
    assert health[1] == "ok"
    latency = m.pool.test_model_latency(ep_obj.base_url, ep_obj.api_key, ep_obj.model,
                                        timeout=10, use_proxy=False, protocol="openai")
    assert latency["ok"] is True
    app_server.shutdown()
    app_server.server_close()
    server.shutdown()
    server.server_close()


def main():
    check("inbound /v1/responses over OpenAI chat upstream (text/image/reasoning/stream)", test_inbound_openai_text)
    check("reasoning effort xhigh normalized for upstream", test_reasoning_xhigh_normalized_for_upstream)
    check("reasoning effort cascade downgrade on HTTP 400", test_reasoning_cascade_downgrade)
    check("reasoning effort clamped by endpoint max", test_max_reasoning_effort_clamp)
    check("single tool call auto serialized on HTTP 400", test_single_tool_call_auto_serialize)
    check("empty assistant messages cleaned before forwarding", test_empty_assistant_message_cleaned)
    check("context overflow trimmed and retried", test_context_overflow_trimmed_and_retried)
    check("context trim helpers", test_trim_context_payload_helpers)
    check("deepseek reasoning echo preserved and backfilled", test_deepseek_reasoning_echo)
    check("OpenAI chat upstream tool calls (non-stream + stream)", test_openai_tool_calls)
    check("Anthropic upstream tool calls (non-stream + stream)", test_anthropic_tool_calls)
    check("native Responses upstream (inbound + chat regression)", test_native_responses_upstream)
    check("Responses upstream chat stream emits finish_reason", test_responses_upstream_chat_stream_finish_reason)
    check("Responses upstream finish without usage stays single-ended", test_responses_upstream_chat_stream_finish_without_usage)
    check("store / previous_response_id / retrieve / delete chain", test_store_previous_reasoning_retrieve_delete)
    check("error cases", test_error_cases)
    check("chat completions regression + health call sites", test_regression_chat_and_health_call_sites)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print(f"\nSUMMARY: {passed} passed, {failed} failed, {len(RESULTS)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
