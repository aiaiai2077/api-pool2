"""Tests for vision downgrade translation: instruction preservation and per-image labels."""

import json
import os
import sys
import threading
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


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
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def request(base, method, path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.read()


class VisionParserMock(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.calls.append(body)
        last_content = body["messages"][-1]["content"]
        url = last_content[-1]["image_url"]
        if isinstance(url, dict):
            url = url.get("url", "")
        desc = "描述A" if "imgA" in str(url) else "描述B"
        data = json.dumps({
            "id": "chatcmpl-vp",
            "object": "chat.completion",
            "model": "vision-parser",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": desc},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class TextTargetMock(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.calls.append(body)
        data = json.dumps({
            "id": "chatcmpl-t",
            "object": "chat.completion",
            "model": "text-target",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_vision_translation_keeps_instruction_and_labels_images():
    VisionParserMock.calls.clear()
    TextTargetMock.calls.clear()
    vision_server, vision_port = start_server(VisionParserMock)
    target_server, target_port = start_server(TextTargetMock)

    m.pool = m.APIPool(default_payload={"temperature": 0.7})
    m.pool.add_endpoint({
        "name": "test_target",
        "base_url": f"http://127.0.0.1:{target_port}",
        "api_key": "sk-test",
        "model": "text-target",
        "priority": 1,
        "timeout": 10,
        "max_retries": 0,
        "enabled": True,
        "use_proxy": False,
        "protocol": "openai",
        "is_vision": False
    })
    m.pool.add_endpoint({
        "name": "test_vision",
        "base_url": f"http://127.0.0.1:{vision_port}",
        "api_key": "sk-test",
        "model": "vision-parser",
        "priority": 2,
        "timeout": 10,
        "max_retries": 0,
        "enabled": True,
        "use_proxy": False,
        "protocol": "openai",
        "is_vision": True
    })

    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, body = request(base, "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "请分别描述这两张截图里的内容"},
            {"type": "image_url", "image_url": "data:image/png;base64,imgA"},
            {"type": "image_url", "image_url": "data:image/png;base64,imgB"}
        ]}]
    })
    assert status == 200
    assert json.loads(body)["choices"][0]["message"]["content"] == "ok"

    sent = TextTargetMock.calls[-1]
    content = sent["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "请分别描述这两张截图里的内容" in content[0]["text"]
    assert "[图片1 解析内容]: 描述A" in content[1]["text"]
    assert "[图片2 解析内容]: 描述B" in content[2]["text"]
    assert all(c["type"] != "image_url" for c in content)

    assert len(VisionParserMock.calls) == 2
    for call in VisionParserMock.calls:
        system = call["messages"][0]["content"]
        assert "结构化 UI 信息" in system and "坐标" in system
        parts = call["messages"][-1]["content"]
        texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
        assert any("请分别描述这两张截图里的内容" in t for t in texts)

    app_server.shutdown()
    app_server.server_close()
    target_server.shutdown()
    target_server.server_close()
    vision_server.shutdown()
    vision_server.server_close()


def test_vision_translation_detects_embedded_input_image_in_tool_text():
    VisionParserMock.calls.clear()
    TextTargetMock.calls.clear()
    vision_server, vision_port = start_server(VisionParserMock)
    target_server, target_port = start_server(TextTargetMock)

    m.pool = m.APIPool(default_payload={"temperature": 0.7})
    m.pool.add_endpoint({
        "name": "test_target",
        "base_url": f"http://127.0.0.1:{target_port}",
        "api_key": "sk-test",
        "model": "text-target",
        "priority": 1,
        "timeout": 10,
        "max_retries": 0,
        "enabled": True,
        "use_proxy": False,
        "protocol": "openai",
        "is_vision": False
    })
    m.pool.add_endpoint({
        "name": "test_vision",
        "base_url": f"http://127.0.0.1:{vision_port}",
        "api_key": "sk-test",
        "model": "vision-parser",
        "priority": 2,
        "timeout": 10,
        "max_retries": 0,
        "enabled": True,
        "use_proxy": False,
        "protocol": "openai",
        "is_vision": True
    })

    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, body = request(base, "POST", "/v1/chat/completions", {
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "tool",
             "content": "[{'type': 'input_image', 'image_url': 'data:image/png;base64,imgA'}]"}
        ]
    })
    assert status == 200
    assert json.loads(body)["choices"][0]["message"]["content"] == "ok"

    sent = TextTargetMock.calls[-1]
    tool_content = sent["messages"][-1]["content"]
    assert "data:image" not in tool_content
    assert "[图片1 解析内容]: 描述A" in tool_content

    assert len(VisionParserMock.calls) == 1
    last_content = VisionParserMock.calls[0]["messages"][-1]["content"]
    assert last_content[-1]["image_url"]["url"] == "data:image/png;base64,imgA"

    app_server.shutdown()
    app_server.server_close()
    target_server.shutdown()
    target_server.server_close()
    vision_server.shutdown()
    vision_server.server_close()


def test_vision_translation_detects_input_image_part():
    VisionParserMock.calls.clear()
    TextTargetMock.calls.clear()
    vision_server, vision_port = start_server(VisionParserMock)
    target_server, target_port = start_server(TextTargetMock)

    m.pool = m.APIPool(default_payload={"temperature": 0.7})
    m.pool.add_endpoint({
        "name": "test_target",
        "base_url": f"http://127.0.0.1:{target_port}",
        "api_key": "sk-test",
        "model": "text-target",
        "priority": 1,
        "timeout": 10,
        "max_retries": 0,
        "enabled": True,
        "use_proxy": False,
        "protocol": "openai",
        "is_vision": False
    })
    m.pool.add_endpoint({
        "name": "test_vision",
        "base_url": f"http://127.0.0.1:{vision_port}",
        "api_key": "sk-test",
        "model": "vision-parser",
        "priority": 2,
        "timeout": 10,
        "max_retries": 0,
        "enabled": True,
        "use_proxy": False,
        "protocol": "openai",
        "is_vision": True
    })

    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, body = request(base, "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "看看"},
            {"type": "input_image", "image_url": "data:image/png;base64,imgB"}
        ]}]
    })
    assert status == 200

    sent = TextTargetMock.calls[-1]
    content = sent["messages"][-1]["content"]
    assert all(c["type"] != "input_image" for c in content)
    assert "[图片1 解析内容]: 描述B" in content[-1]["text"]

    app_server.shutdown()
    app_server.server_close()
    target_server.shutdown()
    target_server.server_close()
    vision_server.shutdown()
    vision_server.server_close()


def test_vision_translation_ignores_historical_images():
    VisionParserMock.calls.clear()
    TextTargetMock.calls.clear()
    vision_server, vision_port = start_server(VisionParserMock)
    target_server, target_port = start_server(TextTargetMock)

    m.pool = m.APIPool(default_payload={"temperature": 0.7})
    m.pool.add_endpoint({
        "name": "test_target",
        "base_url": f"http://127.0.0.1:{target_port}",
        "api_key": "sk-test",
        "model": "text-target",
        "priority": 1,
        "timeout": 10,
        "max_retries": 0,
        "enabled": True,
        "use_proxy": False,
        "protocol": "openai",
        "is_vision": False
    })
    m.pool.add_endpoint({
        "name": "test_vision",
        "base_url": f"http://127.0.0.1:{vision_port}",
        "api_key": "sk-test",
        "model": "vision-parser",
        "priority": 2,
        "timeout": 10,
        "max_retries": 0,
        "enabled": True,
        "use_proxy": False,
        "protocol": "openai",
        "is_vision": True
    })

    app_server = m.ThreadingHTTPServer(("127.0.0.1", 0), m.Handler)
    threading.Thread(target=app_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app_server.server_address[1]}"

    status, body = request(base, "POST", "/v1/chat/completions", {
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "tool",
             "content": "[{'type': 'input_image', 'image_url': 'data:image/png;base64,imgA'}]"},
            {"role": "user", "content": "继续"}
        ]
    })
    assert status == 200
    assert len(VisionParserMock.calls) == 0

    app_server.shutdown()
    app_server.server_close()
    target_server.shutdown()
    target_server.server_close()
    vision_server.shutdown()
    vision_server.server_close()


def main():
    check("vision translation keeps text instruction and labels each image",
          test_vision_translation_keeps_instruction_and_labels_images)
    check("vision translation detects embedded input_image in tool text",
          test_vision_translation_detects_embedded_input_image_in_tool_text)
    check("vision translation detects input_image content part",
          test_vision_translation_detects_input_image_part)
    check("vision translation ignores images from earlier turns",
          test_vision_translation_ignores_historical_images)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print(f"\nSUMMARY: {passed} passed, {failed} failed, {len(RESULTS)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
