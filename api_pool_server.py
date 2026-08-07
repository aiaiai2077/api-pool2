"""
API Pool — 聚合 API 自动切换模块（GUI 版）

启动: python api_pool_server.py
访问: http://localhost:5100
"""

import os
import json
import time
import threading
import sqlite3
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from datetime import datetime, timedelta
from collections import deque

LATENCY_OK_MAX = 2000     
LATENCY_SLOW_MAX = 5000   
HEALTH_CHECK_INTERVAL = 120  

REASONING_EFFORT_MAP = {
    "xhigh": "high",
    "xlow": "low",
    "minimal": "low",
}
REASONING_EFFORT_LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
REASONING_EFFORT_CHAIN = ["xhigh", "high", "medium", "low"]

def _normalize_reasoning_effort(effort, allow_xhigh=False):
    if not isinstance(effort, str):
        return effort
    key = effort.strip().lower()
    if allow_xhigh and key == "xhigh":
        return effort
    return REASONING_EFFORT_MAP.get(key, effort)

def _clamp_reasoning_effort(effort, max_effort):
    if not isinstance(effort, str) or not max_effort:
        return effort
    key = effort.strip().lower()
    max_key = str(max_effort).strip().lower()
    if key not in REASONING_EFFORT_LEVELS or max_key not in REASONING_EFFORT_LEVELS:
        return effort
    if REASONING_EFFORT_LEVELS.index(key) > REASONING_EFFORT_LEVELS.index(max_key):
        return max_key
    return effort

def _downgrade_reasoning_effort(effort):
    if not isinstance(effort, str):
        return None
    key = effort.strip().lower()
    if key == "max":
        return "high"
    if key == "minimal":
        return "low"
    if key == "none":
        return None
    if key in REASONING_EFFORT_CHAIN:
        idx = REASONING_EFFORT_CHAIN.index(key)
        if idx + 1 < len(REASONING_EFFORT_CHAIN):
            return REASONING_EFFORT_CHAIN[idx + 1]
    return None

def _downgrade_reasoning_payload(payload):
    import copy
    p = copy.deepcopy(payload)
    changed = False
    if "reasoning_effort" in p:
        new_val = _downgrade_reasoning_effort(p["reasoning_effort"])
        if new_val is None:
            del p["reasoning_effort"]
        else:
            p["reasoning_effort"] = new_val
        changed = True
    reasoning = p.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        new_val = _downgrade_reasoning_effort(reasoning["effort"])
        if new_val is None:
            del reasoning["effort"]
            if not reasoning:
                del p["reasoning"]
        else:
            reasoning["effort"] = new_val
        changed = True
    return p if changed else None


class LogManager:
    def __init__(self, max_history=300):
        self.history = []
        self.lock = threading.Lock()
        self.max_history = max_history
        self._counter = 0

    def log(self, level, msg):
        ts = time.time()
        time_str = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        with self.lock:
            self._counter += 1
            entry = {"id": self._counter, "time": time_str, "level": level, "msg": msg, "timestamp": ts}
            self.history.append(entry)
            if len(self.history) > self.max_history:
                self.history.pop(0)

    def get_logs_since(self, last_id):
        with self.lock:
            return [log for log in self.history if log["id"] > last_id]

    def clear_logs(self):
        with self.lock:
            self.history.clear()

sys_logger = LogManager()
def sys_log(msg, level="INFO"):
    sys_logger.log(level, msg)
    print(f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}")

class TokenTracker:
    def __init__(self, db_path="token_stats.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON token_usage(timestamp)")
            try:
                conn.execute("ALTER TABLE token_usage ADD COLUMN endpoint_name TEXT DEFAULT ''")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE token_usage ADD COLUMN cached_tokens INTEGER DEFAULT 0")
            except Exception:
                pass

    def add_usage(self, endpoint_name, model, prompt_tokens, completion_tokens, total_tokens, cached_tokens=0):
        def _do_insert():
            try:
                with sqlite3.connect(self.db_path, timeout=5) as conn:
                    conn.execute(
                        "INSERT INTO token_usage (endpoint_name, model, prompt_tokens, completion_tokens, total_tokens, cached_tokens) VALUES (?, ?, ?, ?, ?, ?)",
                        (endpoint_name, model, prompt_tokens, completion_tokens, total_tokens, cached_tokens)
                    )
            except Exception as e:
                sys_log(f"记录 token 消耗失败: {e}", "WARN")
        threading.Thread(target=_do_insert, daemon=True).start()

    def get_today_usage_by_endpoint(self, endpoint_name):
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT SUM(total_tokens) FROM token_usage WHERE endpoint_name = ? AND timestamp >= datetime(date('now', 'localtime'), 'utc')", (endpoint_name,))
                return cursor.fetchone()[0] or 0
        except Exception:
            return 0

    def rename_endpoint(self, old_name: str, new_name: str):
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("UPDATE token_usage SET endpoint_name = ? WHERE endpoint_name = ?", (new_name, old_name))
        except Exception as e:
            sys_log(f"重命名端点统计数据失败: {e}", "WARN")

    def get_stats(self, endpoint_filter=None):
        with sqlite3.connect(self.db_path, timeout=5) as conn:
            cursor = conn.cursor()
            ep_cond = " AND endpoint_name = ?" if endpoint_filter and endpoint_filter != "all" else ""
            params = (endpoint_filter,) if (endpoint_filter and endpoint_filter != "all") else ()
            
            cursor.execute(f"SELECT SUM(total_tokens), SUM(cached_tokens), SUM(prompt_tokens), COUNT(*) FROM token_usage WHERE timestamp >= datetime(date('now', 'localtime'), 'utc'){ep_cond}", params)
            today_row = cursor.fetchone()
            today = today_row[0] or 0
            today_cached = today_row[1] or 0
            today_prompt = today_row[2] or 0
            today_calls = today_row[3] or 0
            today_cache_hit_rate = round(today_cached / today_prompt * 100, 1) if today_prompt > 0 else 0
            
            cursor.execute(f"SELECT SUM(total_tokens) FROM token_usage WHERE timestamp >= datetime(date('now', '-2 days', 'localtime'), 'utc'){ep_cond}", params)
            last_3_days = cursor.fetchone()[0] or 0
            cursor.execute(f"SELECT SUM(total_tokens) FROM token_usage WHERE timestamp >= datetime(date('now', '-6 days', 'localtime'), 'utc'){ep_cond}", params)
            last_7_days = cursor.fetchone()[0] or 0
            cursor.execute(f"SELECT SUM(total_tokens), SUM(cached_tokens), SUM(prompt_tokens), COUNT(*) FROM token_usage WHERE timestamp >= datetime(date('now', '-29 days', 'localtime'), 'utc'){ep_cond}", params)
            month_row = cursor.fetchone()
            last_30_days = month_row[0] or 0
            month_cached = month_row[1] or 0
            month_prompt = month_row[2] or 0
            month_calls = month_row[3] or 0
            month_cache_hit_rate = round(month_cached / month_prompt * 100, 1) if month_prompt > 0 else 0
            
            cursor.execute(f"""
                SELECT date(timestamp, 'localtime') as d, SUM(total_tokens), SUM(prompt_tokens), SUM(cached_tokens), SUM(completion_tokens)
                FROM token_usage
                WHERE timestamp >= datetime(date('now', '-13 days', 'localtime'), 'utc'){ep_cond}
                GROUP BY d
            """, params)
            raw_trend = {r[0]: {"total": r[1] or 0, "prompt": r[2] or 0, "cached": r[3] or 0, "completion": r[4] or 0} for r in cursor.fetchall()}
            trend_14d = []
            now = datetime.now()
            for i in range(13, -1, -1):
                d_str = (now - timedelta(days=i)).strftime('%Y-%m-%d')
                data = raw_trend.get(d_str, {"total": 0, "prompt": 0, "cached": 0, "completion": 0})
                trend_14d.append({"date": d_str, "tokens": data["total"], "prompt": data["prompt"], "cached": data["cached"], "completion": data["completion"]})
                
            cursor.execute(f"""
                SELECT strftime('%H', datetime(timestamp, 'localtime')) as h, SUM(total_tokens), COUNT(*), SUM(prompt_tokens), SUM(cached_tokens)
                FROM token_usage
                WHERE timestamp >= datetime(date('now', 'localtime'), 'utc'){ep_cond}
                GROUP BY h
            """, params)
            raw_hourly = {r[0]: (r[1], r[2], r[3] or 0, r[4] or 0) for r in cursor.fetchall()}
            trend_today_hourly = []
            for i in range(24):
                h_str = f"{i:02d}"
                val = raw_hourly.get(h_str, (0, 0, 0, 0))
                missed = max(0, val[2] - val[3])
                trend_today_hourly.append({"date": f"{h_str}:00", "tokens": val[0] or 0, "calls": val[1] or 0, "missed": missed})
                
            cursor.execute(f"""
                SELECT endpoint_name, model, SUM(total_tokens), COUNT(*), SUM(prompt_tokens), SUM(cached_tokens)
                FROM token_usage
                WHERE timestamp >= datetime(date('now', 'localtime'), 'utc'){ep_cond}
                GROUP BY endpoint_name, model
                ORDER BY SUM(total_tokens) DESC
            """, params)
            today_endpoints = [{"endpoint": r[0] or "未知端点", "model": r[1], "tokens": r[2] or 0, "calls": r[3] or 0, "cache_hit_rate": round((r[5] or 0)/(r[4] or 1)*100, 1) if (r[4] or 0) > 0 else 0} for r in cursor.fetchall()]
            
            cursor.execute(f"""
                SELECT endpoint_name, model, SUM(total_tokens), COUNT(*), SUM(prompt_tokens), SUM(cached_tokens)
                FROM token_usage
                WHERE strftime('%Y-%m', timestamp, 'localtime') = strftime('%Y-%m', 'now', 'localtime'){ep_cond}
                GROUP BY endpoint_name, model
                ORDER BY SUM(total_tokens) DESC
            """, params)
            month_endpoints = [{"endpoint": r[0] or "未知端点", "model": r[1], "tokens": r[2] or 0, "calls": r[3] or 0, "cache_hit_rate": round((r[5] or 0)/(r[4] or 1)*100, 1) if (r[4] or 0) > 0 else 0} for r in cursor.fetchall()]

            cursor.execute("SELECT DISTINCT endpoint_name FROM token_usage WHERE endpoint_name IS NOT NULL")
            all_endpoints_list = [r[0] for r in cursor.fetchall()]

            return {
                "today": today,
                "today_cached": today_cached,
                "today_missed": max(0, today_prompt - today_cached),
                "today_calls": today_calls,
                "today_cache_hit_rate": today_cache_hit_rate,
                "last_3_days": last_3_days,
                "last_7_days": last_7_days,
                "last_30_days": last_30_days,
                "month_cached": month_cached,
                "month_missed": max(0, month_prompt - month_cached),
                "month_calls": month_calls,
                "month_cache_hit_rate": month_cache_hit_rate,
                "trend_14d": trend_14d,
                "trend_today_hourly": trend_today_hourly,
                "today_endpoints": today_endpoints,
                "month_endpoints": month_endpoints,
                "all_endpoints_list": all_endpoints_list
            }

    def export_csv(self):
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Timestamp", "Endpoint", "Model", "Prompt Tokens", "Completion Tokens", "Total Tokens", "Cached Tokens"])
        with sqlite3.connect(self.db_path, timeout=5) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp, endpoint_name, model, prompt_tokens, completion_tokens, total_tokens, cached_tokens FROM token_usage ORDER BY id DESC")
            for row in cursor.fetchall():
                writer.writerow(row)
        return output.getvalue()

    def clear_data(self):
        with sqlite3.connect(self.db_path, timeout=5) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM token_usage")
            conn.commit()

token_tracker = TokenTracker()

class ChatLogger:
    def __init__(self, db_path="chat_logs.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                endpoint_name TEXT,
                model TEXT,
                prompt TEXT,
                completion TEXT,
                total_tokens INTEGER,
                latency_ms INTEGER
            )''')
            conn.commit()
            conn.close()

    def add_log(self, endpoint_name, model, prompt, completion, total_tokens, latency_ms):
        def _write():
            with self._lock:
                try:
                    conn = sqlite3.connect(self.db_path)
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO chat_logs (endpoint_name, model, prompt, completion, total_tokens, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
                        (endpoint_name, model, prompt, completion, total_tokens, latency_ms)
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    sys_log(f"记录对话日志失败: {e}", "ERROR")
        threading.Thread(target=_write, daemon=True).start()

    def get_logs(self, limit=50, offset=0):
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                c.execute(
                    "SELECT id, datetime(timestamp, 'localtime'), endpoint_name, model, prompt, completion, total_tokens, latency_ms FROM chat_logs ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                )
                rows = c.fetchall()
                
                c.execute("SELECT COUNT(*) FROM chat_logs")
                total = c.fetchone()[0]
                conn.close()
                
                return {
                    "total": total,
                    "logs": [
                        {
                            "id": r[0],
                            "timestamp": r[1],
                            "endpoint_name": r[2],
                            "model": r[3],
                            "prompt": r[4],
                            "completion": r[5],
                            "total_tokens": r[6],
                            "latency_ms": r[7]
                        } for r in rows
                    ]
                }
            except Exception as e:
                return {"total": 0, "logs": [], "error": str(e)}

    def clear_logs(self):
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                c.execute("DELETE FROM chat_logs")
                conn.commit()
                conn.close()
            except Exception:
                pass

chat_logger = ChatLogger()

def extract_prompt_text(payload):
    try:
        messages = payload.get("messages", [])
        output = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, str):
                output.append(f"[{role.upper()}]\n{content}")
            elif isinstance(content, list):
                parts = []
                for part in content:
                    ptype = part.get("type", "")
                    if ptype == "text":
                        parts.append(part.get("text", ""))
                    elif ptype == "image_url":
                        parts.append("[Base64 Image Omitted]")
                output.append(f"[{role.upper()}]\n" + "\n".join(parts))
        return "\n\n".join(output)
    except Exception:
        return str(payload)[:2000]


# ============================================================
#  Responses API (inbound adapter)
# ============================================================

def _responses_instructions_text(instructions):
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, list):
        texts = []
        for item in instructions:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and item.get("type") in ("input_text", "text", "output_text"):
                texts.append(item.get("text", ""))
        return "\n".join(t for t in texts if t)
    return str(instructions or "")


def _responses_content_to_chat(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append({"type": "text", "text": c})
                continue
            if not isinstance(c, dict):
                continue
            ctype = c.get("type")
            if ctype in ("input_text", "output_text", "text"):
                parts.append({"type": "text", "text": c.get("text", "")})
            elif ctype in ("input_image", "image_url"):
                raw_url = c.get("image_url")
                if isinstance(raw_url, dict):
                    url = raw_url.get("url", "")
                    detail = c.get("detail") or raw_url.get("detail")
                else:
                    url = raw_url or ""
                    detail = c.get("detail")
                image = {"url": url}
                if detail:
                    image["detail"] = detail
                parts.append({"type": "image_url", "image_url": image})
        if not parts:
            return ""
        if len(parts) == 1 and parts[0]["type"] == "text":
            return parts[0]["text"]
        return parts
    return str(content)


def _responses_content_text(content):
    if isinstance(content, str):
        return content
    texts = []
    if isinstance(content, list):
        for c in content:
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, dict):
                ctype = c.get("type")
                if ctype in ("input_text", "output_text", "text"):
                    texts.append(c.get("text", ""))
    return "\n".join(t for t in texts if t)


def _responses_input_to_messages(input_data, instructions):
    messages = []
    if instructions:
        instr_text = _responses_instructions_text(instructions)
        if instr_text:
            messages.append({"role": "system", "content": instr_text})

    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
        return messages
    if not isinstance(input_data, list):
        raise ValueError("input must be a string or an array of input items")

    user_parts = []

    def flush_user_parts():
        nonlocal user_parts
        if user_parts:
            messages.append({"role": "user", "content": user_parts})
            user_parts = []

    for item in input_data:
        if isinstance(item, str):
            user_parts.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in ("input_text", "text"):
            user_parts.append({"type": "text", "text": item.get("text", "")})
        elif item_type in ("input_image", "image_url"):
            raw_url = item.get("image_url")
            if isinstance(raw_url, dict):
                url = raw_url.get("url", "")
                detail = item.get("detail") or raw_url.get("detail")
            else:
                url = raw_url or ""
                detail = item.get("detail")
            image = {"url": url}
            if detail:
                image["detail"] = detail
            user_parts.append({"type": "image_url", "image_url": image})
        elif item_type == "message":
            role = item.get("role", "user")
            content = _responses_content_to_chat(item.get("content"))
            if role in ("system", "developer"):
                text = content if isinstance(content, str) else _responses_content_text(content)
                if text:
                    messages.append({"role": "system", "content": text})
            else:
                safe_role = role if role in ("user", "assistant", "tool") else "user"
                messages.append({"role": safe_role, "content": content or ""})
        elif item_type == "function_call":
            flush_user_parts()
            fn = item.get("function") or {}
            tool_call = {
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", "") or fn.get("name", ""),
                    "arguments": item.get("arguments", "") or fn.get("arguments", "")
                }
            }
            if messages and messages[-1].get("role") == "assistant":
                messages[-1].setdefault("tool_calls", []).append(tool_call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        elif item_type == "function_call_output":
            flush_user_parts()
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(output)
            })

    if user_parts:
        messages.append({"role": "user", "content": user_parts})
    if not messages:
        raise ValueError("input is empty or contains no supported items")
    return messages


def _responses_tools_to_chat(tools):
    if not isinstance(tools, list):
        return []
    chat_tools = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        chat_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters") or {"type": "object", "properties": {}}
            }
        })
    return chat_tools


def _responses_tool_choice_to_chat(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice.get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return None


def _responses_text_format_to_chat(text_param):
    if not isinstance(text_param, dict):
        return None
    fmt = text_param.get("format")
    if isinstance(fmt, str):
        return {"type": fmt} if fmt in ("text", "json_object") else None
    if not isinstance(fmt, dict):
        return None
    fmt_type = fmt.get("type")
    if fmt_type == "json_schema":
        json_schema = fmt.get("json_schema")
        if not isinstance(json_schema, dict):
            json_schema = fmt
        return {
            "type": "json_schema",
            "json_schema": {
                "name": json_schema.get("name") or "response",
                "schema": json_schema.get("schema") or {},
                "strict": bool(json_schema.get("strict", False))
            }
        }
    if fmt_type in ("text", "json_object"):
        return {"type": fmt_type}
    return None


def _chat_response_format_to_responses_text(response_format):
    if not isinstance(response_format, dict):
        return None
    fmt_type = response_format.get("type")
    if fmt_type == "json_schema":
        json_schema = response_format.get("json_schema") or {}
        return {
            "format": {
                "type": "json_schema",
                "name": json_schema.get("name") or "response",
                "schema": json_schema.get("schema") or {},
                "strict": bool(json_schema.get("strict", False))
            }
        }
    if fmt_type in ("text", "json_object"):
        return {"format": {"type": fmt_type}}
    return None


def _responses_to_chat_request(body):
    input_data = body.get("input")
    if input_data is None:
        raise ValueError("input is required")
    messages = _responses_input_to_messages(input_data, body.get("instructions"))

    extra = {}
    for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty",
                "seed", "user", "metadata", "logprobs", "top_logprobs"):
        if key in body:
            extra[key] = body[key]
    if "max_output_tokens" in body:
        extra["max_tokens"] = body["max_output_tokens"]
    if "max_completion_tokens" in body:
        extra["max_tokens"] = body["max_completion_tokens"]
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        extra["reasoning_effort"] = reasoning["effort"]
    response_format = _responses_text_format_to_chat(body.get("text"))
    if response_format:
        extra["response_format"] = response_format

    tools = _responses_tools_to_chat(body.get("tools"))
    if tools:
        extra["tools"] = tools
        if "parallel_tool_calls" in body:
            extra["parallel_tool_calls"] = bool(body["parallel_tool_calls"])
        tool_choice = _responses_tool_choice_to_chat(body.get("tool_choice"))
        if tool_choice is not None:
            extra["tool_choice"] = tool_choice
    return messages, extra


def _responses_default_usage():
    return {
        "input_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 0
    }


def _responses_usage_from_chat_usage(usage):
    if not usage:
        return _responses_default_usage()
    reasoning_tokens = 0
    if isinstance(usage.get("completion_tokens_details"), dict):
        reasoning_tokens = usage["completion_tokens_details"].get("reasoning_tokens", 0) or 0
    cached_tokens = 0
    if isinstance(usage.get("prompt_tokens_details"), dict):
        cached_tokens = usage["prompt_tokens_details"].get("cached_tokens", 0) or 0
    return {
        "input_tokens": usage.get("prompt_tokens", 0) or 0,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": usage.get("completion_tokens", 0) or 0,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": usage.get("total_tokens", 0) or 0
    }


def _chat_content_to_responses_content(content):
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    parts = []
    for c in content or []:
        if not isinstance(c, dict):
            continue
        ctype = c.get("type")
        if ctype == "text":
            parts.append({"type": "input_text", "text": c.get("text", "")})
        elif ctype == "image_url":
            raw_url = c.get("image_url") or {}
            if isinstance(raw_url, dict):
                url = raw_url.get("url", "")
                detail = c.get("detail") or raw_url.get("detail")
            else:
                url = raw_url or ""
                detail = c.get("detail")
            image = {"url": url}
            if detail:
                image["detail"] = detail
            parts.append({"type": "input_image", "image_url": image})
    return parts


def _chat_messages_to_responses_input(messages):
    instructions = []
    items = []
    for m in messages or []:
        role = m.get("role")
        if role == "system":
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
            if content:
                instructions.append(content)
            continue
        if role == "assistant":
            content = m.get("content") or ""
            if content:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": _chat_content_to_responses_content(content)
                })
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "") or ""
                })
        elif role == "tool":
            output = m.get("content", "")
            if isinstance(output, list):
                output = "".join(p.get("text", "") for p in output if isinstance(p, dict) and p.get("type") == "text")
            items.append({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id", ""),
                "output": output or ""
            })
        else:
            items.append({
                "type": "message",
                "role": "user",
                "content": _chat_content_to_responses_content(m.get("content", ""))
            })
    return items, "\n".join(i for i in instructions if i).strip()


def _responses_tools_from_chat(tools):
    if not isinstance(tools, list):
        return []
    responses_tools = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function") or {}
        responses_tools.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}}
        })
    return responses_tools


def _responses_body_from_chat(payload):
    items, instructions = _chat_messages_to_responses_input(payload.get("messages", []))
    body = {
        "model": payload.get("model", ""),
        "input": items or "",
    }
    if instructions:
        body["instructions"] = instructions
    if "max_tokens" in payload:
        body["max_output_tokens"] = payload["max_tokens"]
    for key in ("stream", "temperature", "top_p", "user", "metadata",
                "parallel_tool_calls", "store", "reasoning", "previous_response_id"):
        if key in payload:
            body[key] = payload[key]
    if "reasoning" not in body and "reasoning_effort" in payload:
        body["reasoning"] = {"effort": payload["reasoning_effort"]}
    if "response_format" in payload:
        text = _chat_response_format_to_responses_text(payload["response_format"])
        if text:
            body["text"] = text
    tools = _responses_tools_from_chat(payload.get("tools"))
    if tools:
        body["tools"] = tools
        tool_choice = payload.get("tool_choice")
        if tool_choice is not None:
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                body["tool_choice"] = {
                    "type": "function",
                    "name": tool_choice.get("function", {}).get("name", "")
                }
            else:
                body["tool_choice"] = tool_choice
    return body


def _responses_output_to_chat_message(output):
    text = ""
    tool_calls = []
    for item in output or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text += part.get("text", "")
        elif item.get("type") == "function_call":
            tool_calls.append({
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "") or ""
                }
            })
    return text, tool_calls


def _responses_usage_to_chat_usage(usage):
    if not usage:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0}
        }
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("input_tokens", 0) or 0,
        "completion_tokens": usage.get("output_tokens", 0) or 0,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "prompt_tokens_details": {"cached_tokens": input_details.get("cached_tokens", 0) or 0},
        "completion_tokens_details": {"reasoning_tokens": output_details.get("reasoning_tokens", 0) or 0}
    }


RESPONSES_DB = "responses_store.db"
RESPONSES_STORE_LOCK = threading.Lock()
RESPONSES_STORE_MAX = 200


def _responses_store_init():
    with sqlite3.connect(RESPONSES_DB, timeout=5) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS responses_store (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.commit()


def _responses_store_put(response_id, data):
    with RESPONSES_STORE_LOCK:
        with sqlite3.connect(RESPONSES_DB, timeout=5) as conn:
            conn.execute(
                """
                INSERT INTO responses_store (id, data, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET data = excluded.data, created_at = excluded.created_at
                """,
                (response_id, json.dumps(data, ensure_ascii=False), time.time())
            )
            conn.execute(
                """
                DELETE FROM responses_store
                WHERE id NOT IN (
                    SELECT id FROM responses_store ORDER BY created_at DESC LIMIT ?
                )
                """,
                (RESPONSES_STORE_MAX,)
            )
            conn.commit()


def _responses_store_get(response_id):
    with RESPONSES_STORE_LOCK:
        with sqlite3.connect(RESPONSES_DB, timeout=5) as conn:
            cursor = conn.execute(
                "SELECT data FROM responses_store WHERE id = ?",
                (response_id,)
            )
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None


def _responses_store_delete(response_id):
    with RESPONSES_STORE_LOCK:
        with sqlite3.connect(RESPONSES_DB, timeout=5) as conn:
            cursor = conn.execute(
                "DELETE FROM responses_store WHERE id = ?",
                (response_id,)
            )
            conn.commit()
            return cursor.rowcount > 0


_responses_store_init()


def _responses_function_call_item(tool_call, idx):
    fn = tool_call.get("function") or {}
    call_id = tool_call.get("id") or f"call_{int(time.time() * 1000)}_{idx}"
    return {
        "id": f"fc_{int(time.time() * 1000)}_{idx}",
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": fn.get("name", ""),
        "arguments": fn.get("arguments", "") or "",
        "output": None
    }


def _responses_response_object(body, text, usage=None, model="api-pool-aggregated",
                               status="completed", response_id=None, item_id=None,
                               message=None, output=None):
    now = int(time.time())
    rid = response_id or f"resp_{int(time.time() * 1000)}"
    mid = item_id or f"msg_{int(time.time() * 1000)}"
    text_value = text if isinstance(text, str) else str(text or "")
    item = {
        "id": mid,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text_value, "annotations": []}]
    }
    if output is not None:
        output_items = output
    else:
        msg = message or {}
        tool_calls = msg.get("tool_calls") or []
        output_items = []
        if text_value or not tool_calls:
            output_items.append(item)
        for idx, tool_call in enumerate(tool_calls):
            output_items.append(_responses_function_call_item(tool_call, idx))
    text_param = body.get("text")
    if isinstance(text_param, dict):
        text_format = text_param.get("format")
        if not isinstance(text_format, dict):
            text_format = {"type": "text"}
    else:
        text_format = {"type": "text"}
    reasoning_param = body.get("reasoning")
    reasoning_effort = reasoning_param.get("effort") if isinstance(reasoning_param, dict) else None
    return {
        "id": rid,
        "object": "response",
        "created_at": now,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "model": model,
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "previous_response_id": body.get("previous_response_id"),
        "reasoning": {"effort": reasoning_effort, "summary": None},
        "store": bool(body.get("store", False)),
        "temperature": body.get("temperature", 0.7),
        "text": {"format": text_format},
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools", []),
        "top_p": body.get("top_p", 1),
        "truncation": body.get("truncation", "disabled"),
        "usage": usage or _responses_default_usage(),
        "user": body.get("user"),
        "metadata": body.get("metadata") or {},
        "expires_at": None,
        "output": output_items
    }


def _responses_sse(event_type, data):
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8")


def _parse_chat_completion_chunk(chunk):
    if not chunk:
        return None
    raw = chunk.decode("utf-8", errors="ignore")
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            return None
        try:
            return json.loads(payload)
        except Exception:
            pass
    return None


def _responses_stream_generator(upstream_gen, body, meta=None, model="api-pool-aggregated",
                                response_id=None, item_id=None, on_complete=None):
    response_id = response_id or f"resp_{int(time.time() * 1000)}"
    item_id = item_id or f"msg_{int(time.time() * 1000)}"
    output_index = 0
    content_index = 0
    final_text = ""
    final_usage = None
    tool_states = {}
    next_output_index = 1
    seq = 0

    def emit(event_type, data):
        nonlocal seq
        seq += 1
        payload = {"sequence_number": seq, "response_id": response_id}
        payload.update(data)
        return _responses_sse(event_type, payload)

    base_response = _responses_response_object(body, "", status="in_progress",
                                               response_id=response_id, item_id=item_id,
                                               model=model)
    base_item = base_response["output"][0]
    base_part = base_item["content"][0]
    output_items = [base_item]

    yield emit("response.created", {"type": "response.created", "response": base_response})
    yield emit("response.in_progress", {"type": "response.in_progress", "response": base_response})
    yield emit("response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": base_item
    })
    yield emit("response.content_part.added", {
        "type": "response.content_part.added",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": base_part
    })

    try:
        for chunk in upstream_gen:
            parsed = _parse_chat_completion_chunk(chunk)
            if not parsed:
                continue
            choices = parsed.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                text_delta = delta.get("content")
                if text_delta:
                    final_text += text_delta
                    yield emit("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "delta": text_delta
                    })
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", len(tool_states))
                    state = tool_states.get(idx)
                    if state is None:
                        call_id = tc.get("id") or f"call_{int(time.time() * 1000)}_{idx}"
                        fc_id = f"fc_{int(time.time() * 1000)}_{idx}"
                        name = (tc.get("function") or {}).get("name", "")
                        state = {
                            "item_id": fc_id,
                            "call_id": call_id,
                            "name": name,
                            "arguments": "",
                            "output_index": next_output_index,
                            "item": {
                                "id": fc_id,
                                "type": "function_call",
                                "status": "in_progress",
                                "call_id": call_id,
                                "name": name,
                                "arguments": "",
                                "output": None
                            }
                        }
                        tool_states[idx] = state
                        output_items.append(state["item"])
                        next_output_index += 1
                        yield emit("response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": state["output_index"],
                            "item": state["item"]
                        })
                    if tc.get("id"):
                        state["call_id"] = tc["id"]
                        state["item"]["call_id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        state["name"] = fn["name"]
                        state["item"]["name"] = fn["name"]
                    arg_delta = fn.get("arguments") or ""
                    if arg_delta:
                        state["arguments"] += arg_delta
                        state["item"]["arguments"] = state["arguments"]
                        yield emit("response.function_call_arguments.delta", {
                            "type": "response.function_call_arguments.delta",
                            "item_id": state["item_id"],
                            "output_index": state["output_index"],
                            "delta": arg_delta
                        })
            if parsed.get("usage"):
                final_usage = parsed["usage"]

        meta_message = (meta or {}).get("message") or {}
        for idx, tc in enumerate((meta_message.get("tool_calls") or [])):
            fn = tc.get("function") or {}
            arguments = fn.get("arguments", "") or ""
            call_id = tc.get("id") or f"call_{int(time.time() * 1000)}_{idx}"
            existing = [s for s in tool_states.values()]
            if idx in tool_states:
                match = tool_states[idx]
            elif call_id:
                match = next((s for s in existing if s["call_id"] == call_id), None)
            else:
                match = next(
                    (s for s in existing if s["name"] == fn.get("name") and not s["call_id"]),
                    None
                )
            if match is not None:
                if arguments and not match["arguments"]:
                    match["arguments"] = arguments
                    match["item"]["arguments"] = arguments
                    yield emit("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "item_id": match["item_id"],
                        "output_index": match["output_index"],
                        "delta": arguments
                    })
                continue
            fc_id = f"fc_{int(time.time() * 1000)}_{idx}"
            state = {
                "item_id": fc_id,
                "call_id": call_id,
                "name": fn.get("name", ""),
                "arguments": arguments,
                "output_index": next_output_index,
                "item": {
                    "id": fc_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": fn.get("name", ""),
                    "arguments": arguments,
                    "output": None
                }
            }
            tool_states[idx] = state
            output_items.append(state["item"])
            next_output_index += 1
            yield emit("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": state["output_index"],
                "item": state["item"]
            })
            if arguments:
                yield emit("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": state["item_id"],
                    "output_index": state["output_index"],
                    "delta": arguments
                })

        yield emit("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "text": final_text
        })
        done_part = {"type": "output_text", "text": final_text, "annotations": []}
        yield emit("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": done_part
        })
        done_item = {
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [done_part]
        }
        yield emit("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": done_item
        })
        completed_output = [done_item]
        for idx in sorted(tool_states, key=lambda i: tool_states[i]["output_index"]):
            state = tool_states[idx]
            yield emit("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": state["item_id"],
                "output_index": state["output_index"],
                "name": state["name"],
                "arguments": state["arguments"]
            })
            done_call = {
                "id": state["item_id"],
                "type": "function_call",
                "status": "completed",
                "call_id": state["call_id"],
                "name": state["name"],
                "arguments": state["arguments"],
                "output": None
            }
            completed_output.append(done_call)
            yield emit("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": state["output_index"],
                "item": done_call
            })
        if final_usage is None and (meta or {}).get("usage"):
            final_usage = meta["usage"]
        completed_response = _responses_response_object(
            body, final_text,
            usage=_responses_usage_from_chat_usage(final_usage),
            status="completed",
            response_id=response_id,
            item_id=item_id,
            model=model,
            output=completed_output
        )
        yield emit("response.completed", {
            "type": "response.completed",
            "response": completed_response
        })
        yield b"data: [DONE]\n\n"
        if on_complete:
            on_complete(completed_response)
    except Exception as e:
        error_obj = {"message": str(e), "type": "server_error", "param": None, "code": None}
        failed_response = _responses_response_object(
            body, final_text,
            usage=_responses_usage_from_chat_usage(final_usage),
            status="failed",
            response_id=response_id,
            item_id=item_id,
            model=model,
            output=output_items
        )
        failed_response["error"] = error_obj
        yield emit("response.failed", {
            "type": "response.failed",
            "response": failed_response
        })
        yield b"data: [DONE]\n\n"


def _anthropic_text_from_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            c.get("text", "") for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        )
    return str(content or "")


def _anthropic_user_content_blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks = []
    for c in content or []:
        if not isinstance(c, dict):
            continue
        ctype = c.get("type")
        if ctype == "text":
            blocks.append({"type": "text", "text": c.get("text", "")})
        elif ctype == "image_url":
            raw_url = c.get("image_url")
            if isinstance(raw_url, dict):
                url = raw_url.get("url", "")
            else:
                url = raw_url or ""
            if url.startswith("data:image/"):
                try:
                    media_type = url.split(";", 1)[0].replace("data:", "")
                    b64_data = url.split(",", 1)[1]
                    blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64_data}
                    })
                except Exception:
                    pass
            else:
                blocks.append({"type": "text", "text": f"[Image URL: {url}]"})
    return blocks or [{"type": "text", "text": ""}]


def _anthropic_assistant_content_blocks(message):
    content = message.get("content") or ""
    blocks = [
        block for block in _anthropic_user_content_blocks(content)
        if not (block.get("type") == "text" and not block.get("text"))
    ]
    for tool_call in message.get("tool_calls") or []:
        fn = tool_call.get("function") or {}
        arguments = fn.get("arguments", "") or ""
        try:
            input_obj = json.loads(arguments) if arguments else {}
        except Exception:
            input_obj = {"raw": arguments} if arguments else {}
        blocks.append({
            "type": "tool_use",
            "id": tool_call.get("id", ""),
            "name": fn.get("name", ""),
            "input": input_obj
        })
    return blocks


def _anthropic_append_message(messages, role, blocks):
    if messages and messages[-1]["role"] == role and isinstance(messages[-1]["content"], list):
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": blocks})


def _anthropic_tools_from_chat(tools):
    if not isinstance(tools, list):
        return []
    anthropic_tools = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function") or {}
        anthropic_tools.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}
        })
    return anthropic_tools


def _anthropic_tool_choice_from_chat(tool_choice):
    if isinstance(tool_choice, str):
        if tool_choice in ("required", "any"):
            return {"type": "any"}
        return {"type": tool_choice if tool_choice in ("auto", "none") else "auto"}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fn = tool_choice.get("function") or {}
        name = fn.get("name") or tool_choice.get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


# ============================================================
#  数据结构
# ============================================================

@dataclass
class Endpoint:
    id: str = ""
    name: str = "unnamed"
    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    priority: int = 999
    timeout: int = 60
    max_retries: int = 1
    enabled: bool = True
    cooldown_minutes: int = 5
    daily_limit: int = 0
    rpm_limit: int = 0
    use_proxy: bool = True
    protocol: str = "openai"
    extra_headers: dict = field(default_factory=dict)
    is_vision: bool = True
    supports_xhigh: bool = False
    max_reasoning_effort: str = ""

    _fail_count: int = field(default=0, repr=False)
    _req_timestamps: deque = field(default_factory=deque, repr=False)
    _rpm_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_error: str = field(default="", repr=False)
    _last_error_ts: float = field(default=0, repr=False)
    _last_success_ts: float = field(default=0, repr=False)
    _total_calls: int = field(default=0, repr=False)
    _total_failures: int = field(default=0, repr=False)
    _cooldown_until: float = field(default=0, repr=False)
    
    _today_used: int = field(default=0, repr=False)
    _today_date: str = field(default="", repr=False)
    health_mode: str = field(default="chat")

    _health: str = field(default="unknown", repr=False) 
    _health_latency_ms: int = field(default=-1, repr=False)
    _health_last_check: float = field(default=0, repr=False)
    _health_error: str = field(default="", repr=False)


class AllEndpointsFailed(Exception):
    def __init__(self, errors: list):
        self.errors = errors
        super().__init__(f"All endpoints failed: {errors}")


# ============================================================
#  API Pool
# ============================================================

class APIPool:
    def __init__(self, endpoints=None, default_payload=None):
        self._lock = threading.Lock()
        self.default_payload = default_payload or {}
        self._endpoints: list[Endpoint] = []
        self._current_idx = 0
        if endpoints:
            for ep in endpoints:
                self.add_endpoint(ep)

    def add_endpoint(self, ep):
        if isinstance(ep, dict):
            ep = Endpoint(**{k: v for k, v in ep.items() if k in Endpoint.__dataclass_fields__})
        if not ep.id:
            import uuid
            ep.id = str(uuid.uuid4())
        ep._today_date = datetime.now().strftime("%Y-%m-%d")
        ep._today_used = token_tracker.get_today_usage_by_endpoint(ep.name)
        with self._lock:
            self._endpoints.append(ep)
            self._endpoints.sort(key=lambda e: e.priority)
            self._current_idx = 0

    def remove_endpoint(self, ep_id):
        with self._lock:
            self._endpoints = [e for e in self._endpoints if e.id != ep_id]
            self._current_idx = 0

    def set_enabled(self, ep_id, enabled):
        with self._lock:
            for ep in self._endpoints:
                if ep.id == ep_id:
                    ep.enabled = enabled
                    break

    def update_endpoint(self, ep_id, updates: dict):
        with self._lock:
            for ep in self._endpoints:
                if ep.id == ep_id:
                    for k, v in updates.items():
                        if hasattr(ep, k) and not k.startswith("_") and k != "id":
                            setattr(ep, k, v)
                    self._endpoints.sort(key=lambda e: e.priority)
                    break

    def list_endpoints(self):
        now = time.time()
        with self._lock:
            active = [ep for ep in self._endpoints if ep.enabled]
            current_ep = active[self._current_idx] if active and self._current_idx < len(active) else None
            return [self._ep_to_dict(ep, ep is current_ep, now) for ep in self._endpoints]

    def _ep_to_dict(self, ep, is_current, now):
        return {
            "id": ep.id,
            "name": ep.name,
            "base_url": ep.base_url,
            "api_key": ep.api_key[:8] + "***" if len(ep.api_key) > 8 else "***",
            "api_key_full": ep.api_key,
            "model": ep.model,
            "priority": ep.priority,
            "timeout": ep.timeout,
            "max_retries": ep.max_retries,
            "enabled": ep.enabled,
            "cooldown_minutes": ep.cooldown_minutes,
            "daily_limit": ep.daily_limit,
            "today_used": ep._today_used,
            "rpm_limit": ep.rpm_limit,
            "use_proxy": ep.use_proxy,
            "protocol": ep.protocol,
            "health_mode": ep.health_mode,
            "is_vision": ep.is_vision,
            "supports_xhigh": ep.supports_xhigh,
            "max_reasoning_effort": ep.max_reasoning_effort,
            "is_rpm_limited": self._is_rpm_limited(ep),
            "fail_count": ep._fail_count,
            "last_error": ep._last_error,
            "last_success": ep._last_success_ts,
            "total_calls": ep._total_calls,
            "total_failures": ep._total_failures,
            "is_current": is_current,
            "in_cooldown": ep._cooldown_until > now,
            "cooldown_remaining": max(0, int(ep._cooldown_until - now)),
            "cooldown_until": ep._cooldown_until,
            "health": ep._health,
            "health_latency_ms": ep._health_latency_ms,
            "health_last_check": ep._health_last_check,
            "health_error": ep._health_error,
        }

    def get_active_chain(self):
        now = time.time()
        with self._lock:
            active = self._active_endpoints()
            current_ep = active[self._current_idx] if active and self._current_idx < len(active) else None
            return [
                {
                    "name": ep.name,
                    "model": ep.model,
                    "priority": ep.priority,
                    "is_current": ep is current_ep,
                    "fail_count": ep._fail_count,
                    "in_cooldown": ep._cooldown_until > now,
                    "cooldown_remaining": max(0, int(ep._cooldown_until - now)),
                    "daily_limit": ep.daily_limit,
                    "today_used": ep._today_used,
                    "rpm_limit": ep.rpm_limit,
                    "use_proxy": ep.use_proxy,
                    "is_rpm_limited": self._is_rpm_limited(ep),
                    "is_vision": ep.is_vision,
                    "supports_xhigh": ep.supports_xhigh,
                    "max_reasoning_effort": ep.max_reasoning_effort,
                    "health": ep._health,
                    "health_latency_ms": ep._health_latency_ms,
                    "health_error": ep._health_error,
                }
                for ep in active
            ]

    def reset(self):
        with self._lock:
            for ep in self._endpoints:
                ep._fail_count = 0
                ep._last_error = ""
                ep._last_error_ts = 0
                ep._cooldown_until = 0
                with ep._rpm_lock:
                    ep._req_timestamps.clear()
            self._current_idx = 0

    def _check_one_health(self, ep):
        if ep.health_mode == "none":
            return ep.id, "unknown", -1, "已禁用健康检测"
            
        if ep.health_mode == "models":
            t0 = time.time()
            try:
                models = self.fetch_models(ep.base_url, ep.api_key, timeout=10, use_proxy=ep.use_proxy, protocol=ep.protocol)
                latency = int((time.time() - t0) * 1000)
                if models:
                    return ep.id, "ok", latency, ""
                else:
                    return ep.id, "bad", latency, "获取模型列表失败"
            except Exception as e:
                return ep.id, "bad", int((time.time() - t0) * 1000), f"Models接口错误: {e}"[:100]
                
        payload = {"model": ep.model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 3}
        
        # Attempt 1
        t0 = time.time()
        reply, err, _ = self._try_endpoint(ep, payload, timeout=10, log_usage=False, force_no_retry=True)
        latency = int((time.time() - t0) * 1000)
        
        if reply is not None and latency <= LATENCY_OK_MAX:
            return ep.id, "ok", latency, ""
            
        # Evaluate if we should retry
        err_str = err[:100] if err else ""
        hard_errors = ["auth error", "400", "401", "403", "404", "429"]
        if any(code in err_str for code in hard_errors):
            return ep.id, "bad", latency, err_str
            
        # Attempt 2 (Retry for cold start or transient glitch)
        t1 = time.time()
        reply2, err2, _ = self._try_endpoint(ep, payload, timeout=10, log_usage=False, force_no_retry=True)
        latency2 = int((time.time() - t1) * 1000)
        
        if reply2 is not None and latency2 <= LATENCY_OK_MAX:
            return ep.id, "ok", latency2, ""
            
        # If retry also fails or isn't fast enough, return the original attempt's status
        if reply is not None:
            if latency <= LATENCY_SLOW_MAX:
                return ep.id, "slow", latency, ""
            else:
                return ep.id, "bad", latency, f"延迟过高: {latency}ms"
        else:
            return ep.id, "bad", latency, err_str or "未知错误"

    def _has_images(self, messages):
        if not messages:
            return False
        last_user = -1
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                last_user = i
        for m in (messages[last_user:] if last_user >= 0 else messages):
            content = m.get("content")
            if isinstance(content, list):
                for c in content:
                    if c.get("type") in ("image_url", "input_image"):
                        return True
            elif isinstance(content, str) and "data:image/" in content:
                return True
        return False

    def _translate_images_sync(self, messages, active_eps):
        vision_eps = [e for e in active_eps if getattr(e, "is_vision", True)]
        if not vision_eps:
            return messages

        sys_prompt = (
            "你是一个专业图像解析器。请结合用户针对图片提出的问题，"
            "将图片内容转化为极其详细的文字描述（包括画面细节、OCR文字、代码片段等）。"
            "如果图片是软件界面、网页、控制台、桌面或手机截图，请额外输出结构化 UI 信息："
            "估算视口尺寸（像素，左上角为原点），并按从左上到右下的顺序列出每个可见元素："
            "类型（按钮/输入框/链接/图标/文本/弹窗/菜单/图片）、完整可见文字或 OCR、中心坐标与宽高（像素）。"
            "只输出对该图片的描述或回答，不要有多余的客套话。"
        )

        import copy
        import re
        data_url_re = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+")

        tasks = []
        task_index = {}

        def add_task(msg_idx, kind, key, accompanying, raw_url, ref):
            task_index[(msg_idx, kind, key)] = len(tasks)
            tasks.append((msg_idx, kind, key, accompanying, raw_url, ref))

        last_user = -1
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                last_user = i

        for msg_idx, m in enumerate(messages):
            if msg_idx < last_user:
                continue
            content = m.get("content")
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                accompanying = "\n".join(t for t in text_parts if t).strip()
                for part_idx, c in enumerate(content):
                    if c.get("type") not in ("image_url", "input_image"):
                        continue
                    raw = c.get("image_url")
                    if isinstance(raw, dict):
                        url = raw.get("url", "")
                        detail = c.get("detail") or raw.get("detail")
                    else:
                        url = raw or ""
                        detail = c.get("detail")
                    image = {"url": url}
                    if detail:
                        image["detail"] = detail
                    add_task(msg_idx, "part", (msg_idx, part_idx), accompanying, url, {
                        "part_idx": part_idx,
                        "image": image,
                    })
            elif isinstance(content, str) and "data:image/" in content:
                spans = []
                for dm in data_url_re.finditer(content):
                    start, end = dm.start(), dm.end()
                    lb = content.rfind("{", 0, start)
                    if lb != -1:
                        rb = content.find("}", end)
                        if rb != -1:
                            block = content[lb:rb + 1]
                            if block.count("{") == block.count("}") and "input_image" in block:
                                start, end = lb, rb + 1
                    spans.append((start, end, dm.group(0)))
                if not spans:
                    continue
                instruction = content
                for start, end, _ in sorted(spans, reverse=True):
                    instruction = instruction[:start] + instruction[end:]
                instruction = instruction.strip()
                for span_idx, (start, end, raw_url) in enumerate(spans):
                    add_task(msg_idx, "span", (msg_idx, span_idx), instruction, raw_url, {
                        "start": start,
                        "end": end,
                    })

        if not tasks:
            return messages

        desc_cache = {}
        descriptions = {}
        for task_idx, (msg_idx, kind, key, accompanying, raw_url, ref) in enumerate(tasks):
            if raw_url in desc_cache:
                descriptions[task_idx] = desc_cache[raw_url]
                continue
            user_content = []
            if accompanying:
                user_content.append({"type": "text", "text": accompanying})
            user_content.append({"type": "image_url", "image_url": {"url": raw_url}})
            parse_msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ]
            description = ""
            for v_ep in vision_eps:
                sys_log(f"启动图片解析 ({task_idx + 1}/{len(tasks)}) -> 尝试端点 {v_ep.name} ({v_ep.model})", "INFO")
                payload = {"model": v_ep.model, "messages": parse_msgs, "stream": False, "max_tokens": 4096}
                result, error, _ = self._try_endpoint(v_ep, payload, timeout=60, log_usage=True, force_no_retry=True)
                if error:
                    sys_log(f"图片解析失败 ({v_ep.name} - {v_ep.model}): {error}", "WARNING")
                    continue
                description = result if isinstance(result, str) else result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if description:
                    break
            if description:
                descriptions[task_idx] = description
                desc_cache[raw_url] = description

        if not descriptions:
            sys_log("所有图片解析端点均失败", "ERROR")
            return messages

        new_msgs = copy.deepcopy(messages)
        img_no = 0
        for msg_idx, m in enumerate(new_msgs):
            content = m.get("content")
            if isinstance(content, list):
                new_content = []
                for part_idx, c in enumerate(content):
                    if c.get("type") not in ("image_url", "input_image"):
                        new_content.append(c)
                        continue
                    task_idx = task_index.get((msg_idx, "part", (msg_idx, part_idx)))
                    img_no += 1
                    desc = descriptions.get(task_idx)
                    if desc:
                        new_content.append({"type": "text", "text": f"\n\n[图片{img_no} 解析内容]: {desc}"})
                    else:
                        new_content.append({"type": "text", "text": f"\n\n[图片{img_no} 解析失败]"})
                m["content"] = new_content
            elif isinstance(content, str):
                spans = [ref for (mi, kind, key, acc, raw, ref) in tasks if mi == msg_idx and kind == "span"]
                if not spans:
                    continue
                pieces = []
                pos = 0
                for idx, ref in enumerate(sorted(spans, key=lambda r: r["start"])):
                    task_idx = task_index.get((msg_idx, "span", (msg_idx, idx)))
                    img_no += 1
                    desc = descriptions.get(task_idx)
                    label = f"[图片{img_no} 解析内容]: {desc}" if desc else f"[图片{img_no} 解析失败]"
                    pieces.append(content[pos:ref["start"]])
                    pieces.append(label)
                    pos = ref["end"]
                pieces.append(content[pos:])
                m["content"] = "".join(pieces)
        sys_log(f"图片解析完成，共解析 {len(descriptions)}/{len(tasks)} 张", "INFO")
        return new_msgs

    def check_all_health(self):
        with self._lock:
            endpoints = [ep for ep in self._endpoints if ep.enabled]
            for ep in endpoints:
                ep._health = "testing"
        if not endpoints:
            return []
        results = []
        with ThreadPoolExecutor(max_workers=min(len(endpoints), 10)) as pool_exec:
            futures = {pool_exec.submit(self._check_one_health, ep): ep for ep in endpoints}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    ep = futures[future]
                    results.append((ep.id, "bad", -1, str(e)))
        now = time.time()
        with self._lock:
            id_map = {ep.id: ep for ep in self._endpoints}
            for ep_id, health, latency, error in results:
                ep = id_map.get(ep_id)
                if ep:
                    ep._health = health
                    ep._health_latency_ms = latency
                    ep._health_last_check = now
                    ep._health_error = error
        sys_log(f"健康检测完成: 检测了 {len(endpoints)} 个端点", "INFO")
        return [{"id": i, "health": h, "latency_ms": l, "error": e} for i, h, l, e in results]

    def _is_in_cooldown(self, ep):
        return ep._cooldown_until > time.time()

    def _is_quota_exceeded(self, ep):
        if ep.daily_limit <= 0: return False
        now_date = datetime.now().strftime("%Y-%m-%d")
        if ep._today_date != now_date:
            ep._today_date = now_date
            ep._today_used = 0
        return ep._today_used >= ep.daily_limit

    def _is_rpm_limited(self, ep):
        if ep.rpm_limit <= 0: return False
        now = time.time()
        with ep._rpm_lock:
            while ep._req_timestamps and ep._req_timestamps[0] < now - 60:
                ep._req_timestamps.popleft()
            return len(ep._req_timestamps) >= ep.rpm_limit

    def _set_cooldown(self, ep):
        if ep.cooldown_minutes > 0:
            ep._cooldown_until = time.time() + ep.cooldown_minutes * 60

    def _clear_cooldown(self, ep):
        ep._cooldown_until = 0

    def _active_endpoints(self):
        available = [ep for ep in self._endpoints if ep.enabled and not self._is_in_cooldown(ep) and not self._is_quota_exceeded(ep) and not self._is_rpm_limited(ep)]
        if available:
            return available
        return [ep for ep in self._endpoints if ep.enabled and not self._is_quota_exceeded(ep)]

    def _pick_best(self, active):
        for ep in active:
            if not self._is_in_cooldown(ep):
                return ep
        return min(active, key=lambda e: e._cooldown_until) if active else None

    def _rotate(self, failed_ep, error_msg):
        failed_ep._fail_count += 1
        failed_ep._total_failures += 1
        failed_ep._last_error = error_msg
        failed_ep._last_error_ts = time.time()
        self._set_cooldown(failed_ep)
        sys_log(f"端点 '{failed_ep.name}' 触发冷却机制，下次可用时间在 {failed_ep.cooldown_minutes} 分钟后", "WARN")
        active = self._active_endpoints()
        if active:
            for i, ep in enumerate(active):
                if ep is failed_ep:
                    self._current_idx = (i + 1) % len(active)
                    return
            self._current_idx = 0

    def _on_success(self, ep):
        ep._total_calls += 1
        ep._last_success_ts = time.time()
        ep._fail_count = 0
        ep._last_error = ""
        self._clear_cooldown(ep)
        active = self._active_endpoints()
        best = self._pick_best(active)
        if best and best.priority < ep.priority:
            for i, e in enumerate(active):
                if e is best:
                    self._current_idx = i
                    return
        for i, e in enumerate(active):
            if e is ep:
                self._current_idx = i
                return

    def chat(self, messages, model=None, extra_payload=None, timeout=None, return_endpoint=False, return_meta=False):
        active = self._active_endpoints()
        if not active:
            raise ValueError("没有可用的 API 端点")
        errors = []
        tried = 0
        total = len(active)
        with self._lock:
            if self._current_idx >= total:
                self._current_idx = 0
            idx = self._current_idx
        while tried < total:
            ep = active[idx]
            ep_timeout = timeout or ep.timeout
            ep_model = model or ep.model
            payload = {
                "model": ep_model, "messages": messages,
                **self.default_payload, **(extra_payload or {}),
            }
            if "reasoning_effort" in payload:
                payload["reasoning_effort"] = _clamp_reasoning_effort(
                    payload["reasoning_effort"], getattr(ep, "max_reasoning_effort", "")
                )
                payload["reasoning_effort"] = _normalize_reasoning_effort(
                    payload["reasoning_effort"], getattr(ep, "supports_xhigh", False)
                )
            reasoning = payload.get("reasoning")
            if isinstance(reasoning, dict) and "effort" in reasoning:
                reasoning["effort"] = _clamp_reasoning_effort(
                    reasoning["effort"], getattr(ep, "max_reasoning_effort", "")
                )
                reasoning["effort"] = _normalize_reasoning_effort(
                    reasoning["effort"], getattr(ep, "supports_xhigh", False)
                )
            
            # [VISION TRANSLATION INTERCEPT]
            if self._has_images(payload["messages"]) and getattr(ep, "is_vision", True) is False:
                has_vision = any(getattr(e, "is_vision", True) for e in active)
                if has_vision:
                    if payload.get("stream"):
                        meta = {}

                        def vision_wrapper(tgt_ep, pld, t_out, a_eps):
                            translated_msgs = self._translate_images_sync(pld["messages"], a_eps)
                            pld["messages"] = translated_msgs
                            gen, err, inner_meta = self._try_endpoint(tgt_ep, pld, t_out)
                            if err:
                                yield f"data: {{'choices':[{{'delta':{{'content':'\\n\\n[API Pool Error: 请求最终目标失败: {err}]'}}}}]}}\n\n".replace("'", '"').encode("utf-8")
                            else:
                                yield from gen
                            meta["usage"] = inner_meta.get("usage")
                            meta["message"] = inner_meta.get("message")
                        with self._lock:
                            self._on_success(ep)
                        wrapper = vision_wrapper(ep, payload, ep_timeout, active)
                        if return_endpoint:
                            if return_meta: return wrapper, ep, meta
                            return wrapper, ep
                        return wrapper
                    else:
                        payload["messages"] = self._translate_images_sync(payload["messages"], active)
            
            if tried == 0:
                sys_log(f"收到 API 请求，尝试请求端点 '{ep.name}' (模型: {ep_model})", "INFO")
            else:
                sys_log(f"重试请求，尝试端点 '{ep.name}' (模型: {ep_model})", "INFO")

            result, error, meta = self._try_endpoint(ep, payload, ep_timeout)
            if result is not None:
                with self._lock:
                    self._on_success(ep)
                sys_log(f"端点 '{ep.name}' 请求成功 (延迟: 正常)", "INFO")
                if return_endpoint:
                    if return_meta: return result, ep, meta
                    return result, ep
                return result
            errors.append(f"[{ep.name}] {error}")
            sys_log(f"端点 '{ep.name}' 请求失败: {error}", "ERROR")
            with self._lock:
                self._rotate(ep, error)
                active = self._active_endpoints()
                total = len(active)
                if total == 0:
                    break
                next_ep = None
                for i, e in enumerate(active):
                    if e is not ep and not self._is_in_cooldown(e):
                        next_ep = e
                        idx = i
                        break
                if next_ep is None:
                    for i, e in enumerate(active):
                        if e is ep:
                            idx = (i + 1) % len(active)
                            break
                tried += 1
        raise AllEndpointsFailed(errors)

    def _try_endpoint(self, ep, payload, timeout, log_usage=True, force_no_retry=False):
        req_t0 = time.time()
        prompt_text_to_log = extract_prompt_text(payload) if log_usage and not ep.name.startswith("test_") else ""
        meta = {"usage": None, "message": None}
        is_anthropic = (getattr(ep, "protocol", "openai") == "anthropic")
        is_responses = (getattr(ep, "protocol", "openai") == "responses")
        
        if is_responses:
            url = ep.base_url.rstrip("/") + "/responses"
            data = json.dumps(_responses_body_from_chat(payload)).encode("utf-8")
        elif is_anthropic:
            base = ep.base_url.rstrip("/")
            if "api.anthropic.com" in base and not base.endswith("/v1"):
                url = base + "/v1/messages"
            else:
                url = base + "/messages"
            anthropic_payload = {
                "model": payload.get("model", ep.model),
                "max_tokens": payload.get("max_tokens", 4096),
            }
            if "temperature" in payload: anthropic_payload["temperature"] = payload["temperature"]
            if "top_p" in payload: anthropic_payload["top_p"] = payload["top_p"]
            if "top_k" in payload: anthropic_payload["top_k"] = payload["top_k"]
            if "stream" in payload: anthropic_payload["stream"] = payload["stream"]
            
            sys_prompt = ""
            messages = []
            for m in payload.get("messages", []):
                role = m.get("role")
                if role == "system":
                    sys_prompt += _anthropic_text_from_content(m.get("content")) + "\n"
                elif role == "assistant":
                    _anthropic_append_message(
                        messages, "assistant", _anthropic_assistant_content_blocks(m)
                    )
                elif role == "tool":
                    output = _anthropic_text_from_content(m.get("content"))
                    _anthropic_append_message(messages, "user", [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": output
                    }])
                else:
                    _anthropic_append_message(
                        messages, "user", _anthropic_user_content_blocks(m.get("content"))
                    )
            if sys_prompt:
                anthropic_payload["system"] = sys_prompt.strip()
            anthropic_payload["messages"] = messages
            anthropic_tools = _anthropic_tools_from_chat(payload.get("tools"))
            if anthropic_tools:
                anthropic_payload["tools"] = anthropic_tools
                tool_choice = _anthropic_tool_choice_from_chat(payload.get("tool_choice"))
                if tool_choice:
                    anthropic_payload["tool_choice"] = tool_choice
            if "stop" in payload:
                stop = payload["stop"]
                anthropic_payload["stop_sequences"] = stop if isinstance(stop, list) else [stop]
            if "user" in payload:
                anthropic_payload["metadata"] = {"user_id": str(payload["user"])}
            data = json.dumps(anthropic_payload).encode("utf-8")
        else:
            url = ep.base_url.rstrip("/") + "/chat/completions"
            chat_payload = {
                k: v for k, v in payload.items()
                if k not in ("store", "previous_response_id", "metadata")
            }
            data = json.dumps(chat_payload).encode("utf-8")
            
        is_stream = payload.get("stream", False)
        
        retries = 0 if force_no_retry else ep.max_retries
        for attempt in range(retries + 1):
            if ep.rpm_limit > 0:
                with ep._rpm_lock:
                    ep._req_timestamps.append(time.time())
                    
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            if is_anthropic:
                safe_api_key = ep.api_key.encode('ascii', 'ignore').decode('ascii').strip()
                req.add_header("x-api-key", safe_api_key)
                req.add_header("Authorization", f"Bearer {safe_api_key}")
                req.add_header("anthropic-version", "2023-06-01")
            else:
                safe_api_key = ep.api_key.encode('ascii', 'ignore').decode('ascii').strip()
                req.add_header("Authorization", f"Bearer {safe_api_key}")
                
            for k, v in ep.extra_headers.items():
                req.add_header(k, v)
                
            try:
                if getattr(ep, "use_proxy", True) is False:
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    resp = opener.open(req, timeout=timeout)
                else:
                    resp = urllib.request.urlopen(req, timeout=timeout)
                
                if is_stream:
                    def stream_generator():
                        stream_id = f"chatcmpl-{int(time.time()*1000)}"
                        final_prompt_tokens = 0
                        final_completion_tokens = 0
                        final_total_tokens = 0
                        final_cached_tokens = 0
                        has_usage = False
                        final_completion_text = ""
                        tool_calls_state = {}
                        anthropic_tool_blocks = {}
                        responses_tool_states = {}
                        done_sent = False
                        anthropic_stop_reason = None

                        def finish_chunk(reason):
                            return b"data: " + json.dumps({
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": ep.model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]
                            }).encode("utf-8") + b"\n\n"

                        try:
                            for line in resp:
                                if is_anthropic:
                                    if not line.strip() or not line.startswith(b"data: "):
                                        continue
                                    if line.startswith(b"data: [DONE]"):
                                        continue
                                    try:
                                        chunk = json.loads(line[6:].decode("utf-8"))
                                        ctype = chunk.get("type")
                                        if ctype == "content_block_start":
                                            block = chunk.get("content_block", {})
                                            if block.get("type") == "tool_use":
                                                idx = chunk.get("index", len(anthropic_tool_blocks))
                                                anthropic_tool_blocks[idx] = {
                                                    "id": block.get("id", ""),
                                                    "name": block.get("name", ""),
                                                    "arguments": ""
                                                }
                                        elif ctype == "content_block_delta":
                                            delta = chunk.get("delta", {})
                                            if delta.get("type") == "input_json_delta":
                                                idx = chunk.get("index", 0)
                                                if idx in anthropic_tool_blocks:
                                                    anthropic_tool_blocks[idx]["arguments"] += delta.get("partial_json", "")
                                            else:
                                                text = delta.get("text", "")
                                                final_completion_text += text
                                                if text:
                                                    o_chunk = {
                                                        "id": stream_id,
                                                        "object": "chat.completion.chunk",
                                                        "created": int(time.time()),
                                                        "model": ep.model,
                                                        "choices": [{"index": 0, "delta": {"content": text}}]
                                                    }
                                                    yield b"data: " + json.dumps(o_chunk).encode("utf-8") + b"\n\n"
                                        elif ctype == "content_block_stop":
                                            idx = chunk.get("index", 0)
                                            block = anthropic_tool_blocks.get(idx)
                                            if block:
                                                o_chunk = {
                                                    "id": stream_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": int(time.time()),
                                                    "model": ep.model,
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": {
                                                            "tool_calls": [{
                                                                "index": idx,
                                                                "id": block["id"],
                                                                "type": "function",
                                                                "function": {
                                                                    "name": block["name"],
                                                                    "arguments": block["arguments"]
                                                                }
                                                            }]
                                                        },
                                                        "finish_reason": "tool_calls"
                                                    }]
                                                }
                                                yield b"data: " + json.dumps(o_chunk).encode("utf-8") + b"\n\n"
                                        elif ctype == "message_stop":
                                            finish_reason = "tool_calls" if anthropic_tool_blocks else {
                                                "end_turn": "stop",
                                                "stop_sequence": "stop",
                                                "max_tokens": "length",
                                                "tool_use": "tool_calls",
                                            }.get(anthropic_stop_reason, "stop")
                                            yield finish_chunk(finish_reason)
                                            usage_chunk = {
                                                "id": stream_id,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": ep.model,
                                                "choices": [],
                                                "usage": {
                                                    "prompt_tokens": final_prompt_tokens,
                                                    "completion_tokens": final_completion_tokens,
                                                    "total_tokens": final_total_tokens
                                                }
                                            }
                                            yield b"data: " + json.dumps(usage_chunk).encode("utf-8") + b"\n\n"
                                            yield b"data: [DONE]\n\n"
                                            done_sent = True
                                        elif ctype == "message_delta" and "usage" in chunk:
                                            u = chunk["usage"]
                                            final_completion_tokens += u.get("output_tokens", 0)
                                            final_total_tokens += u.get("output_tokens", 0)
                                            has_usage = True
                                            if isinstance(chunk.get("delta"), dict) and chunk["delta"].get("stop_reason"):
                                                anthropic_stop_reason = chunk["delta"]["stop_reason"]
                                        elif ctype == "message_start" and "message" in chunk and "usage" in chunk["message"]:
                                            u = chunk["message"]["usage"]
                                            prompt_t = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                                            final_prompt_tokens += prompt_t
                                            final_total_tokens += prompt_t
                                            final_cached_tokens += u.get("cache_read_input_tokens", 0)
                                            has_usage = True
                                    except Exception:
                                        pass
                                elif is_responses:
                                    if not line.strip() or not line.startswith(b"data: "):
                                        continue
                                    payload_line = line[6:].strip()
                                    if payload_line == b"[DONE]":
                                        continue
                                    try:
                                        evt = json.loads(payload_line.decode("utf-8"))
                                    except Exception:
                                        continue
                                    etype = evt.get("type")
                                    if etype == "response.output_text.delta":
                                        text = evt.get("delta", "")
                                        final_completion_text += text
                                        if text:
                                            o_chunk = {
                                                "id": stream_id,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": ep.model,
                                                "choices": [{"index": 0, "delta": {"content": text}}]
                                            }
                                            yield b"data: " + json.dumps(o_chunk).encode("utf-8") + b"\n\n"
                                    elif etype == "response.output_item.added":
                                        item = evt.get("item") or {}
                                        if item.get("type") == "function_call":
                                            idx = evt.get("output_index", len(responses_tool_states))
                                            responses_tool_states[idx] = {
                                                "id": item.get("call_id", ""),
                                                "name": item.get("name", ""),
                                                "arguments": item.get("arguments", "") or "",
                                                "emitted": False
                                            }
                                    elif etype == "response.function_call_arguments.delta":
                                        idx = evt.get("output_index", len(responses_tool_states))
                                        state = responses_tool_states.setdefault(idx, {
                                            "id": "", "name": "", "arguments": "", "emitted": False
                                        })
                                        state["arguments"] += evt.get("delta", "")
                                    elif etype == "response.function_call_arguments.done":
                                        idx = evt.get("output_index", len(responses_tool_states))
                                        state = responses_tool_states.setdefault(idx, {
                                            "id": "", "name": "", "arguments": "", "emitted": False
                                        })
                                        state["id"] = evt.get("call_id", state["id"])
                                        state["name"] = evt.get("name", state["name"])
                                        state["arguments"] = evt.get("arguments", state["arguments"])
                                        if not state["emitted"]:
                                            state["emitted"] = True
                                            o_chunk = {
                                                "id": stream_id,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": ep.model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {
                                                        "tool_calls": [{
                                                            "index": idx,
                                                            "id": state["id"],
                                                            "type": "function",
                                                            "function": {
                                                                "name": state["name"],
                                                                "arguments": state["arguments"]
                                                            }
                                                        }]
                                                    },
                                                    "finish_reason": "tool_calls"
                                                }]
                                            }
                                            yield b"data: " + json.dumps(o_chunk).encode("utf-8") + b"\n\n"
                                    elif etype == "response.output_item.done":
                                        item = evt.get("item") or {}
                                        if item.get("type") == "function_call":
                                            idx = evt.get("output_index", len(responses_tool_states))
                                            state = responses_tool_states.setdefault(idx, {
                                                "id": "", "name": "", "arguments": "", "emitted": False
                                            })
                                            state["id"] = item.get("call_id", state["id"])
                                            state["name"] = item.get("name", state["name"])
                                            state["arguments"] = item.get("arguments", state["arguments"])
                                            if not state["emitted"]:
                                                state["emitted"] = True
                                                o_chunk = {
                                                    "id": stream_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": int(time.time()),
                                                    "model": ep.model,
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": {
                                                            "tool_calls": [{
                                                                "index": idx,
                                                                "id": state["id"],
                                                                "type": "function",
                                                                "function": {
                                                                    "name": state["name"],
                                                                    "arguments": state["arguments"]
                                                                }
                                                            }]
                                                        },
                                                        "finish_reason": "tool_calls"
                                                    }]
                                                }
                                                yield b"data: " + json.dumps(o_chunk).encode("utf-8") + b"\n\n"
                                    elif etype == "response.completed":
                                        response_obj = evt.get("response") or {}
                                        incomplete = response_obj.get("incomplete_details") or {}
                                        finish_reason = (
                                            "length"
                                            if response_obj.get("status") == "incomplete"
                                            or incomplete.get("reason") == "max_output_tokens"
                                            else "stop"
                                        )
                                        if not responses_tool_states:
                                            yield finish_chunk(finish_reason)
                                        u = response_obj.get("usage") or {}
                                        if u:
                                            final_prompt_tokens = u.get("input_tokens", 0)
                                            final_cached_tokens = (u.get("input_tokens_details") or {}).get("cached_tokens", 0)
                                            final_completion_tokens = u.get("output_tokens", 0)
                                            final_total_tokens = u.get("total_tokens", 0)
                                            has_usage = True
                                            usage_chunk = {
                                                "id": stream_id,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": ep.model,
                                                "choices": [],
                                                "usage": {
                                                    "prompt_tokens": final_prompt_tokens,
                                                    "completion_tokens": final_completion_tokens,
                                                    "total_tokens": final_total_tokens
                                                }
                                            }
                                            yield b"data: " + json.dumps(usage_chunk).encode("utf-8") + b"\n\n"
                                        yield b"data: [DONE]\n\n"
                                        done_sent = True
                                        break
                                else:
                                    yield line
                                    if line.strip() and line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
                                        try:
                                            chunk = json.loads(line[6:].decode("utf-8"))
                                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                                delta = chunk["choices"][0].get("delta", {})
                                                if "content" in delta:
                                                    final_completion_text += delta.get("content", "")
                                                for tc in delta.get("tool_calls") or []:
                                                    idx = tc.get("index", len(tool_calls_state))
                                                    state = tool_calls_state.setdefault(idx, {
                                                        "id": "", "name": "", "arguments": ""
                                                    })
                                                    if tc.get("id"):
                                                        state["id"] = tc["id"]
                                                    fn = tc.get("function") or {}
                                                    if fn.get("name"):
                                                        state["name"] = fn["name"]
                                                    if fn.get("arguments"):
                                                        state["arguments"] += fn.get("arguments", "")
                                            if "usage" in chunk and chunk["usage"]:
                                                u = chunk["usage"]
                                                final_prompt_tokens = u.get("prompt_tokens", 0)
                                                final_completion_tokens = u.get("completion_tokens", 0)
                                                final_total_tokens = u.get("total_tokens", 0)
                                                if "prompt_tokens_details" in u and isinstance(u["prompt_tokens_details"], dict):
                                                    final_cached_tokens = u["prompt_tokens_details"].get("cached_tokens", 0)
                                                has_usage = True
                                        except Exception:
                                            pass
                            if is_responses and not done_sent:
                                if not responses_tool_states:
                                    yield finish_chunk("stop")
                                yield b"data: [DONE]\n\n"
                        except Exception:
                            pass
                        finally:
                            tool_calls = []
                            for idx in sorted(tool_calls_state):
                                state = tool_calls_state[idx]
                                if state["id"] or state["name"]:
                                    tool_calls.append({
                                        "id": state["id"],
                                        "type": "function",
                                        "function": {
                                            "name": state["name"],
                                            "arguments": state["arguments"]
                                        }
                                    })
                            for idx in sorted(anthropic_tool_blocks):
                                block = anthropic_tool_blocks[idx]
                                tool_calls.append({
                                    "id": block["id"],
                                    "type": "function",
                                    "function": {
                                        "name": block["name"],
                                        "arguments": block["arguments"]
                                    }
                                })
                            for idx in sorted(responses_tool_states):
                                state = responses_tool_states[idx]
                                tool_calls.append({
                                    "id": state["id"],
                                    "type": "function",
                                    "function": {
                                        "name": state["name"],
                                        "arguments": state["arguments"]
                                    }
                                })
                            meta["message"] = {
                                "role": "assistant",
                                "content": final_completion_text,
                                "tool_calls": tool_calls or None
                            }
                            meta["usage"] = {
                                "prompt_tokens": final_prompt_tokens,
                                "completion_tokens": final_completion_tokens,
                                "total_tokens": final_total_tokens,
                                "prompt_tokens_details": {"cached_tokens": final_cached_tokens},
                                "completion_tokens_details": {"reasoning_tokens": 0}
                            }
                            if has_usage and log_usage and not ep.name.startswith("test_"):
                                token_tracker.add_usage(ep.name, ep.model, final_prompt_tokens, final_completion_tokens, final_total_tokens, final_cached_tokens)
                                chat_logger.add_log(ep.name, ep.model, prompt_text_to_log, final_completion_text, final_total_tokens, int((time.time() - req_t0) * 1000))
                                ep._today_used += final_total_tokens
                            resp.close()
                    return stream_generator(), "", meta
                else:
                    body = json.loads(resp.read().decode("utf-8"))
                    if is_anthropic:
                        reply = ""
                        tool_calls = []
                        for c in body.get("content", []):
                            if c.get("type") == "text":
                                reply += c.get("text", "")
                            elif c.get("type") == "tool_use":
                                tool_calls.append({
                                    "id": c.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": c.get("name", ""),
                                        "arguments": json.dumps(c.get("input", {}), ensure_ascii=False)
                                    }
                                })
                        u = body.get("usage", {})
                        prompt_t = 0
                        tot = 0
                        cached = 0
                        if u:
                            prompt_t = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                            tot = prompt_t + u.get("output_tokens", 0)
                            cached = u.get("cache_read_input_tokens", 0)
                            if log_usage and not ep.name.startswith("test_"):
                                token_tracker.add_usage(ep.name, ep.model, prompt_t, u.get("output_tokens", 0), tot, cached)
                                chat_logger.add_log(ep.name, ep.model, prompt_text_to_log, reply.strip(), tot, int((time.time() - req_t0) * 1000))
                                ep._today_used += tot
                        meta["message"] = {
                            "role": "assistant",
                            "content": reply.strip(),
                            "tool_calls": tool_calls or None
                        }
                        meta["usage"] = {
                            "prompt_tokens": prompt_t,
                            "completion_tokens": u.get("output_tokens", 0),
                            "total_tokens": tot,
                            "prompt_tokens_details": {"cached_tokens": cached},
                            "completion_tokens_details": {"reasoning_tokens": 0}
                        }
                        return reply.strip(), "", meta
                    elif is_responses:
                        text, tool_calls = _responses_output_to_chat_message(body.get("output", []))
                        u = body.get("usage") or {}
                        chat_usage = _responses_usage_to_chat_usage(u)
                        if log_usage and not ep.name.startswith("test_"):
                            token_tracker.add_usage(
                                ep.name, ep.model,
                                chat_usage["prompt_tokens"],
                                chat_usage["completion_tokens"],
                                chat_usage["total_tokens"],
                                chat_usage["prompt_tokens_details"].get("cached_tokens", 0)
                            )
                            chat_logger.add_log(
                                ep.name, ep.model,
                                prompt_text_to_log,
                                text.strip(),
                                chat_usage["total_tokens"],
                                int((time.time() - req_t0) * 1000)
                            )
                            ep._today_used += chat_usage["total_tokens"]
                        meta["message"] = {
                            "role": "assistant",
                            "content": text.strip(),
                            "tool_calls": tool_calls or None
                        }
                        meta["usage"] = chat_usage
                        return text.strip(), "", meta
                    else:
                        u = body.get("usage", {})
                        tot = 0
                        cached = 0
                        if u:
                            tot = u.get("total_tokens", 0)
                            cached = 0
                            if "prompt_tokens_details" in u and isinstance(u["prompt_tokens_details"], dict):
                                cached = u["prompt_tokens_details"].get("cached_tokens", 0)
                            if log_usage and not ep.name.startswith("test_"):
                                token_tracker.add_usage(ep.name, ep.model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), tot, cached)
                                content = body["choices"][0]["message"].get("content", "")
                                chat_logger.add_log(ep.name, ep.model, prompt_text_to_log, content.strip(), tot, int((time.time() - req_t0) * 1000))
                                ep._today_used += tot
                        message = body["choices"][0]["message"]
                        content = message.get("content", "")
                        meta["message"] = message
                        meta["usage"] = {
                            "prompt_tokens": u.get("prompt_tokens", 0) if u else 0,
                            "completion_tokens": u.get("completion_tokens", 0) if u else 0,
                            "total_tokens": tot if u else 0,
                            "prompt_tokens_details": {"cached_tokens": cached},
                            "completion_tokens_details": u.get("completion_tokens_details", {}) if u else {}
                        }
                        return (content.strip() if content else ""), "", meta
                    
                    
            except urllib.error.HTTPError as e:
                err_body = ""
                try: err_body = e.read().decode("utf-8", errors="ignore")[:200]
                except Exception: pass
                msg = f"HTTP {e.code}: {err_body}"
                if (e.code == 400 and not force_no_retry
                        and any(k in payload for k in ("temperature", "top_p"))
                        and ("temperature" in err_body or "top_p" in err_body)):
                    cleaned = {k: v for k, v in payload.items() if k not in ("temperature", "top_p")}
                    sys_log(f"\u7aef\u70b9 '{ep.name}' \u4e0d\u652f\u6301 temperature/top_p\uff0c\u5df2\u81ea\u52a8\u79fb\u9664\u540e\u91cd\u8bd5", "WARNING")
                    return self._try_endpoint(ep, cleaned, timeout, log_usage=log_usage, force_no_retry=True)
                if e.code == 400 and "reasoning" in err_body.lower():
                    downgraded = _downgrade_reasoning_payload(payload)
                    if downgraded is not None:
                        sys_log(f"\u7aef\u70b9 '{ep.name}' \u4e0d\u652f\u6301\u5f53\u524d reasoning effort\uff0c\u5df2\u81ea\u52a8\u964d\u7ea7\u540e\u91cd\u8bd5", "WARNING")
                        return self._try_endpoint(ep, downgraded, timeout, log_usage=log_usage, force_no_retry=True)
                if e.code == 429: return None, msg + " (429 rate-limited)", meta
                if e.code in (401, 403): return None, msg + " (auth error)", meta
                if e.code >= 500:
                    if attempt < retries:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    return None, msg, meta
                return None, msg, meta
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                msg = f"连接/超时错误: {e}"
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return None, msg, meta
            except Exception as e:
                return None, f"未知错误: {e}", meta
        return None, "重试次数用尽", meta

    def fetch_models(self, base_url, api_key, timeout=10, use_proxy=True, protocol="openai"):
        base = base_url.rstrip("/")
        if protocol == "anthropic" and "api.anthropic.com" in base and not base.endswith("/v1"):
            url = base + "/v1/models"
        else:
            url = base + "/models"
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        safe_api_key = api_key.encode('ascii', 'ignore').decode('ascii').strip()
        
        if protocol == "anthropic":
            req.add_header("x-api-key", safe_api_key)
            req.add_header("Authorization", f"Bearer {safe_api_key}")
            req.add_header("anthropic-version", "2023-06-01")
        else:
            req.add_header("Authorization", f"Bearer {safe_api_key}")
            
        try:
            if not use_proxy:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                resp = opener.open(req, timeout=timeout)
            else:
                resp = urllib.request.urlopen(req, timeout=timeout)
                
            with resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw = data.get("data", [])
                models = []
                for m in raw:
                    mid = m.get("id", "")
                    if not mid: continue
                    info = {"id": mid}
                    if "pricing" in m: info["pricing"] = m["pricing"]
                    if "description" in m: info["description"] = m["description"][:120]
                    info["modality"] = "unknown"
                    info["modality_source"] = "none"
                    models.append(info)
                models.sort(key=lambda x: x["id"])
                return models
        except Exception as e:
            if protocol == "anthropic" and isinstance(e, urllib.error.HTTPError) and e.code == 404:
                raise Exception("该端点尚未支持获取模型列表 (官方老版协议或部分代理不支持)")
            raise e

    def test_vision(self, base_url, api_key, model, timeout=15, use_proxy=True, protocol="openai"):
        ep = Endpoint(name="test_vision", base_url=base_url, api_key=api_key, model=model, max_retries=0, use_proxy=use_proxy, protocol=protocol)
        tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "describe this image in 3 words"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tiny_png}"}}]}],
            "max_tokens": 10,
        }
        t0 = time.time()
        reply, err, _ = self._try_endpoint(ep, payload, timeout)
        latency = int((time.time() - t0) * 1000)
        
        if reply is not None:
            if isinstance(reply, dict):
                reply_text = reply.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
            else:
                reply_text = str(reply).lower()
            unsupported_keywords = ["cannot see", "can't see", "not able to see", "unable to see", "text-based", "language model", "无法查看", "无法读取", "无法看到", "不具备", "不支持", "抱歉", "sorry", "没有上传", "没上传"]
            if any(k in reply_text for k in unsupported_keywords):
                return {"ok": True, "supports_vision": False, "latency_ms": latency, "reply": reply, "error": f"模型疑似无法读图: {reply_text[:50]}..."}
            return {"ok": True, "supports_vision": True, "latency_ms": latency, "reply": reply, "error": ""}
        else:
            unsupported = "image" in err.lower() or "vision" in err.lower() or "content" in err.lower() or "400" in err
            return {"ok": not unsupported, "supports_vision": not unsupported, "latency_ms": latency, "reply": "", "error": err}

    def test_model_latency(self, base_url, api_key, model, timeout=15, use_proxy=True, protocol="openai"):
        ep = Endpoint(name="test_latency", base_url=base_url, api_key=api_key, model=model, max_retries=0, use_proxy=use_proxy, protocol=protocol)
        payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
        t0 = time.time()
        reply, err, _ = self._try_endpoint(ep, payload, timeout)
        latency = int((time.time() - t0) * 1000)
        
        if reply is not None:
            status = "ok" if latency <= LATENCY_OK_MAX else ("slow" if latency <= LATENCY_SLOW_MAX else "bad")
            return {"ok": True, "status": status, "latency_ms": latency, "reply": reply, "error": ""}
        else:
            return {"ok": False, "status": "bad", "latency_ms": latency, "reply": "", "error": err}

CONFIG_FILE = "api_config.json"

def load_config():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f: return json.load(f).get("api_endpoints", [])
    except Exception:
        return []

def save_config(endpoints_data):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"api_endpoints": endpoints_data}, f, ensure_ascii=False, indent=2)

def ensure_config():
    if not os.path.exists(CONFIG_FILE): save_config([])

ensure_config()

def _health_check_loop():
    while True:
        time.sleep(HEALTH_CHECK_INTERVAL)
        try: pool.check_all_health()
        except Exception: pass

pool = APIPool(default_payload={"temperature": 0.7})
for ep_data in load_config(): pool.add_endpoint(ep_data)

_health_thread = threading.Thread(target=_health_check_loop, daemon=True)
_health_thread.start()
threading.Thread(target=pool.check_all_health, daemon=True).start()


def api_handler(method, path, body):
    parsed = urlparse(path)
    cp = parsed.path

    # ================= 代理接口 =================
    if method == "GET" and (cp.startswith("/v1/responses/") or cp.startswith("/responses/")):
        rid = unquote(cp.rsplit("/", 1)[-1])
        stored = _responses_store_get(rid)
        if stored:
            return 200, stored.get("response", stored), False
        return 404, {"error": {"message": "Response not found", "type": "invalid_request_error", "param": None, "code": None}}, False

    if method == "DELETE" and (cp.startswith("/v1/responses/") or cp.startswith("/responses/")):
        rid = unquote(cp.rsplit("/", 1)[-1])
        if _responses_store_delete(rid):
            return 200, {"deleted": True, "id": rid}, False
        return 404, {"error": {"message": "Response not found", "type": "invalid_request_error", "param": None, "code": None}}, False

    if method == "POST" and cp in ("/v1/responses", "/responses"):
        return _handle_responses(body)

    if method == "POST" and cp in ("/v1/chat/completions", "/chat/completions"):
        messages = body.get("messages", [])
        is_stream = body.get("stream", False)
        
        if is_stream:
            if "stream_options" not in body:
                body["stream_options"] = {"include_usage": True}
            elif isinstance(body["stream_options"], dict):
                body["stream_options"]["include_usage"] = True
                
        extra_payload = {k: v for k, v in body.items() if k not in ("messages", "model")}
        
        try:
            if is_stream:
                result = pool.chat(messages, extra_payload=extra_payload)
                return 200, result, True
            result, served_ep, meta = pool.chat(
                messages,
                extra_payload=extra_payload,
                return_endpoint=True,
                return_meta=True
            )
            message = meta.get("message") or {"role": "assistant", "content": result}
            usage = meta.get("usage") or {}
            finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
            
            response = {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": served_ep.model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "prompt_tokens_details": usage.get("prompt_tokens_details") or {},
                    "completion_tokens_details": usage.get("completion_tokens_details") or {}
                }
            }
            return 200, response, False
            
        except AllEndpointsFailed as e:
            return 500, {"error": {"message": f"所有端点均已失效: {e.errors}", "type": "server_error"}}, False
        except Exception as e:
            return 500, {"error": {"message": str(e), "type": "server_error"}}, False

    if method == "GET" and cp == "/api/logs":
        qs = dict(q.split("=") for q in parsed.query.split("&") if "=" in q) if parsed.query else {}
        last_id = int(qs.get("since", 0))
        return 200, sys_logger.get_logs_since(last_id), False

    if method == "DELETE" and cp == "/api/logs":
        sys_logger.clear_logs()
        return 200, {"ok": True}, False

    if method == "GET" and cp == "/api/chat-logs":
        qs = dict(q.split("=") for q in parsed.query.split("&") if "=" in q) if parsed.query else {}
        limit = int(qs.get("limit", 50))
        offset = int(qs.get("offset", 0))
        return 200, chat_logger.get_logs(limit=limit, offset=offset), False

    if method == "DELETE" and cp == "/api/chat-logs":
        chat_logger.clear_logs()
        return 200, {"ok": True}, False

    if method == "GET" and cp == "/api/token-stats":
        qs = dict(q.split("=") for q in parsed.query.split("&") if "=" in q) if parsed.query else {}
        ep = qs.get("endpoint", "all")
        # url decode
        ep = urllib.parse.unquote(ep)
        return 200, token_tracker.get_stats(endpoint_filter=ep), False

    if method == "DELETE" and cp == "/api/token-stats":
        token_tracker.clear_data()
        return 200, {"ok": True}, False

    if method == "GET" and cp == "/api/endpoints": return 200, pool.list_endpoints(), False
    if method == "GET" and cp == "/api/chain": return 200, pool.get_active_chain(), False
    if method == "POST" and cp == "/api/endpoints":
        pool.add_endpoint(body); _sync_to_config(); return 201, {"ok": True}, False
    if method == "POST" and cp == "/api/endpoints/batch":
        items = body.get("endpoints", []); base = body.get("base", {}); added = 0; start_priority = base.get("start_priority", 1)
        for i, item in enumerate(items):
            ep = {
                "name": item.get("name", base.get("name", f"ep_{i}")), "base_url": item.get("base_url", base.get("base_url", "")),
                "api_key": item.get("api_key", base.get("api_key", "")), "model": item.get("model", ""),
                "priority": item.get("priority", start_priority + i), "timeout": item.get("timeout", base.get("timeout", 60)),
                "max_retries": item.get("max_retries", base.get("max_retries", 1)), "cooldown_minutes": item.get("cooldown_minutes", base.get("cooldown_minutes", 5)),
                "daily_limit": item.get("daily_limit", base.get("daily_limit", 0)), "rpm_limit": item.get("rpm_limit", base.get("rpm_limit", 0)),
                "use_proxy": item.get("use_proxy", base.get("use_proxy", True)),
                "protocol": item.get("protocol", base.get("protocol", "openai")),
                "health_mode": item.get("health_mode", base.get("health_mode", "chat")),
                  "is_vision": item.get("is_vision", base.get("is_vision", True)),
                  "supports_xhigh": item.get("supports_xhigh", base.get("supports_xhigh", False)),
                  "max_reasoning_effort": item.get("max_reasoning_effort", base.get("max_reasoning_effort", "")),
                  "enabled": item.get("enabled", True),
            }
            if ep["model"]: pool.add_endpoint(ep); added += 1
        _sync_to_config(); return 201, {"ok": True, "added": added}, False
    if method == "PUT" and cp.startswith("/api/endpoints/") and not cp.endswith("/toggle"):
        ep_id = unquote(cp.split("/")[-1])
        new_name = body.get("name")
        old_ep = next((e for e in pool.list_endpoints() if e["id"] == ep_id), None)
        if old_ep and new_name and new_name != old_ep["name"]:
            token_tracker.rename_endpoint(old_ep["name"], new_name)
        pool.update_endpoint(ep_id, body); _sync_to_config(); return 200, {"ok": True}, False
    if method == "DELETE" and cp.startswith("/api/endpoints/"):
        ep_id = unquote(cp.split("/")[-1]); pool.remove_endpoint(ep_id); _sync_to_config(); return 200, {"ok": True}, False
    if method == "POST" and cp.endswith("/toggle"):
        ep_id = unquote(cp.split("/")[3])
        for ep in pool.list_endpoints():
            if ep["id"] == ep_id: pool.set_enabled(ep_id, not ep["enabled"]); break
        _sync_to_config(); return 200, {"ok": True}, False
    if method == "POST" and cp == "/api/health-check": return 200, {"ok": True, "results": pool.check_all_health()}, False
    if method == "POST" and cp == "/api/fetch-models":
        base_url = body.get("base_url", ""); api_key = body.get("api_key", "")
        if not base_url or not api_key: return 400, {"error": "需要 base_url 和 api_key"}, False
        try:
            models = pool.fetch_models(base_url, api_key, use_proxy=body.get("use_proxy", True), protocol=body.get("protocol", "openai"))
            return 200, {"ok": True, "models": models, "count": len(models)}, False
        except urllib.error.HTTPError as e:
            err_body = ""
            try: err_body = e.read().decode("utf-8", errors="ignore")[:200]
            except Exception: pass
            return 200, {"ok": False, "error": f"HTTP {e.code}: {err_body}"}, False
        except Exception as e: return 200, {"ok": False, "error": str(e)}, False
    if method == "POST" and cp == "/api/test-model": return 200, pool.test_model_latency(body.get("base_url", ""), body.get("api_key", ""), body.get("model", ""), timeout=body.get("timeout", 60), use_proxy=body.get("use_proxy", True), protocol=body.get("protocol", "openai")), False
    if method == "POST" and cp == "/api/test-vision": return 200, pool.test_vision(body.get("base_url", ""), body.get("api_key", ""), body.get("model", ""), timeout=body.get("timeout", 60), use_proxy=body.get("use_proxy", True), protocol=body.get("protocol", "openai")), False
    if method == "POST" and cp == "/api/test":
        ep_id = body.get("id", ""); test_msg = body.get("message", "你好"); target_ep = None
        for ep in pool.list_endpoints():
            if ep["id"] == ep_id: target_ep = ep; break
        if not target_ep: return 404, {"error": "端点不存在"}, False
        test_pool = APIPool(default_payload={"temperature": 0.7})
        test_pool.add_endpoint({"name": target_ep["name"], "base_url": target_ep["base_url"], "api_key": target_ep["api_key_full"], "model": target_ep["model"], "priority": 1, "timeout": target_ep["timeout"], "max_retries": target_ep["max_retries"], "enabled": True, "use_proxy": target_ep.get("use_proxy", True), "protocol": target_ep.get("protocol", "openai"), "is_vision": target_ep.get("is_vision", True)})
        
        img = body.get("image")
        if img:
            test_msg = [{"type": "text", "text": test_msg}, {"type": "image_url", "image_url": {"url": img}}]
            
        try:
            res_dict, served_ep = test_pool.chat([{"role": "user", "content": test_msg}], return_endpoint=True)
            res_str = res_dict.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(res_dict, dict) else res_dict
            return 200, {"ok": True, "result": res_str, "served_by": f"{served_ep.name} ({served_ep.model})"}, False
        except Exception as e: return 200, {"ok": False, "error": str(e)}, False
    if method == "POST" and cp == "/api/test-pool":
        test_msg = body.get("message", "你好")
        img = body.get("image")
        if img:
            test_msg = [{"type": "text", "text": test_msg}, {"type": "image_url", "image_url": {"url": img}}]
        try:
            res_dict, served_ep = pool.chat([{"role": "user", "content": test_msg}], return_endpoint=True)
            res_str = res_dict.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(res_dict, dict) else res_dict
            return 200, {"ok": True, "result": res_str, "served_by": f"{served_ep.name} ({served_ep.model})"}, False
        except AllEndpointsFailed as e: return 200, {"ok": False, "errors": e.errors}, False
        except Exception as e: return 200, {"ok": False, "error": str(e)}, False
    if method == "POST" and cp == "/api/reset": pool.reset(); return 200, {"ok": True}, False

    return 404, {"error": "Not found"}, False


def _handle_responses(body):
    try:
        messages, extra_payload = _responses_to_chat_request(body)
    except ValueError as e:
        return 400, {"error": {"message": str(e), "type": "invalid_request_error", "param": None, "code": None}}, False

    previous_response_id = body.get("previous_response_id")
    if previous_response_id:
        stored = _responses_store_get(previous_response_id)
        if stored:
            stored_messages = stored.get("messages") or []
            if stored_messages:
                messages = stored_messages + messages
        else:
            return 400, {"error": {
                "message": f"previous_response_id '{previous_response_id}' not found",
                "type": "invalid_request_error",
                "param": "previous_response_id",
                "code": None
            }}, False

    is_stream = bool(body.get("stream", False))
    extra_payload["stream"] = is_stream
    if is_stream:
        if "stream_options" not in extra_payload:
            extra_payload["stream_options"] = {"include_usage": True}
        elif isinstance(extra_payload["stream_options"], dict):
            extra_payload["stream_options"]["include_usage"] = True

    response_id = f"resp_{int(time.time() * 1000)}"
    item_id = f"msg_{int(time.time() * 1000)}"

    def maybe_store(response_obj):
        if body.get("store"):
            _responses_store_put(response_obj.get("id") or response_id, {
                "messages": messages,
                "response": response_obj
            })

    try:
        result, served_ep, meta = pool.chat(
            messages,
            extra_payload=extra_payload,
            return_endpoint=True,
            return_meta=True
        )
        if is_stream:
            return 200, _responses_stream_generator(
                result, body,
                meta=meta,
                model=served_ep.model,
                response_id=response_id,
                item_id=item_id,
                on_complete=maybe_store
            ), True
        text = result if isinstance(result, str) else str(result or "")
        response_obj = _responses_response_object(
            body, text,
            usage=_responses_usage_from_chat_usage(meta.get("usage")),
            model=served_ep.model,
            message=meta.get("message"),
            response_id=response_id,
            item_id=item_id
        )
        maybe_store(response_obj)
        return 200, response_obj, False
    except AllEndpointsFailed as e:
        return 500, {"error": {"message": f"所有端点均已失败: {e.errors}", "type": "server_error", "param": None, "code": None}}, False
    except Exception as e:
        return 500, {"error": {"message": str(e), "type": "server_error", "param": None, "code": None}}, False


def _sync_to_config():
    save_config([{"id": ep.get("id"), "name": ep["name"], "base_url": ep["base_url"], "api_key": ep.get("api_key_full", ep.get("api_key", "")), "model": ep["model"], "priority": ep["priority"], "timeout": ep["timeout"], "max_retries": ep["max_retries"], "enabled": ep["enabled"], "cooldown_minutes": ep["cooldown_minutes"], "daily_limit": ep.get("daily_limit", 0), "rpm_limit": ep.get("rpm_limit", 0), "use_proxy": ep.get("use_proxy", True), "protocol": ep.get("protocol", "openai"), "health_mode": ep.get("health_mode", "chat"), "is_vision": ep.get("is_vision", True), "supports_xhigh": ep.get("supports_xhigh", False), "max_reasoning_effort": ep.get("max_reasoning_effort", "")} for ep in pool.list_endpoints()])


GUI_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>API Pool 聚合管理</title>
<style>
:root{
  --bg:#090a0f;--card:rgba(255,255,255,0.03);--card-hover:rgba(255,255,255,0.06);--border:rgba(255,255,255,0.08);
  --text:#ffffff;--text-dim:rgba(255,255,255,0.5);--accent:#5e5ce6;--accent-light:#7d7aff;
  --green:#32d74b;--green-dim:rgba(50,215,75,0.15);--red:#ff453a;--red-dim:rgba(255,69,58,0.15);
  --yellow:#ffd60a;--yellow-dim:rgba(255,214,10,0.15);--blue:#0a84ff;--blue-dim:rgba(10,132,255,0.15);
  --radius:16px;--shadow:0 8px 32px 0 rgba(0,0,0,0.3);--glass-blur:blur(24px);
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text','Inter',system-ui,sans-serif;background-color:var(--bg);background-image:radial-gradient(circle at 15% 50%, rgba(94,92,230,0.2), transparent 50%),radial-gradient(circle at 85% 30%, rgba(10,132,255,0.2), transparent 50%),radial-gradient(circle at 50% 80%, rgba(255,69,58,0.15), transparent 50%);background-attachment:fixed;color:var(--text);min-height:100vh;padding:20px 24px;font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}

.header{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;flex-wrap:wrap;gap:12px}
.header h1{font-size:20px;font-weight:700;letter-spacing:-.3px;display:flex;align-items:center;gap:10px}
.header h1 .logo{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,var(--accent),var(--blue));display:flex;align-items:center;justify-content:center;font-size:16px}
.header-actions{display:flex;gap:8px;flex-wrap:wrap}

.btn{padding:7px 14px;border:none;border-radius:7px;font-size:12px;font-weight:600;cursor:pointer;transition:all .12s;display:inline-flex;align-items:center;gap:5px;letter-spacing:.2px}
.btn:hover{transform:translateY(-1px);filter:brightness(1.1)}
.btn:active{transform:translateY(0)}
.btn-primary{background:var(--accent);color:#fff}
.btn-green{background:var(--green);color:#000}
.btn-red{background:var(--red);color:#fff}
.btn-yellow{background:var(--yellow);color:#000}
.btn-ghost{background:transparent;color:var(--text-dim);border:1px solid var(--border)}
.btn-ghost:hover{border-color:var(--accent);color:var(--accent-light)}
.btn-sm{padding:4px 10px;font-size:11px;border-radius:6px}
.btn:disabled{opacity:.35;cursor:not-allowed;transform:none}

.api-info-card {
  background: rgba(94, 92, 230, 0.08); backdrop-filter: var(--glass-blur); -webkit-backdrop-filter: var(--glass-blur);
  border: 1px solid rgba(94, 92, 230, 0.3); border-radius: var(--radius); padding: 14px 18px; margin-bottom: 20px; box-shadow: var(--shadow);
}
.api-info-card code {
  background: var(--bg);
  padding: 3px 8px;
  border-radius: 4px;
  font-family: 'SF Mono', Menlo, Consolas, monospace;
  font-size: 12px;
  color: var(--accent-light);
  user-select: all;
  border: 1px solid var(--border);
}

.grid{display:grid;grid-template-columns:1fr 360px;gap:20px;align-items:start}
@media(max-width:920px){.grid{grid-template-columns:1fr}}

.card{background:var(--card);backdrop-filter:var(--glass-blur);-webkit-backdrop-filter:var(--glass-blur);border:1px solid var(--border);border-radius:var(--radius);padding:16px 18px;box-shadow:var(--shadow)}
.card-title{font-size:13px;font-weight:700;margin-bottom:14px;display:flex;align-items:center;gap:7px;color:var(--text-dim);text-transform:uppercase;letter-spacing:.6px}
.card-title .icon{font-size:15px}

.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px}
.stat-item{background:rgba(255,255,255,.02);backdrop-filter:var(--glass-blur);-webkit-backdrop-filter:var(--glass-blur);border:1px solid var(--border);border-radius:12px;padding:12px 10px;text-align:center;transition:transform .2s,box-shadow .2s;box-shadow:0 2px 8px rgba(0,0,0,.1)}
.stat-item:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,.2)}
.stat-item .num{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
.stat-item .label{font-size:10px;color:var(--text-dim);margin-top:2px;text-transform:uppercase;letter-spacing:.5px}

.dash-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
.dash-stat{position:relative;overflow:hidden;background:linear-gradient(145deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);backdrop-filter:var(--glass-blur);-webkit-backdrop-filter:var(--glass-blur);border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:20px;text-align:left;transition:transform .3s cubic-bezier(0.2,0.8,0.2,1),box-shadow .3s;box-shadow:0 4px 16px rgba(0,0,0,0.2)}
.dash-stat::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);opacity:0;transition:opacity .3s}
.dash-stat:hover{transform:translateY(-3px);box-shadow:0 8px 24px rgba(0,0,0,0.3);border-color:rgba(255,255,255,0.12)}
.dash-stat:hover::before{opacity:1}
.dash-stat .stat-icon{position:absolute;right:15px;top:15px;font-size:24px;opacity:0.2;transition:opacity .3s, transform .3s}
.dash-stat:hover .stat-icon{opacity:0.4;transform:scale(1.1)}
.dash-stat .num{font-size:28px;font-weight:800;font-variant-numeric:tabular-nums;margin-bottom:4px;letter-spacing:-0.5px}
.dash-stat .label{font-size:11px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px;font-weight:600}

.tbl-progress-container{position:relative;width:100%;height:100%;display:flex;align-items:center}
.tbl-progress-bar{position:absolute;left:0;top:0;bottom:0;background:rgba(94,92,230,0.15);border-radius:4px;z-index:0;transition:width 0.5s cubic-bezier(0.2,0.8,0.2,1)}
.tbl-content{position:relative;z-index:1;padding:6px;width:100%;display:flex;justify-content:space-between;align-items:center}


.filter-bar{display:flex;gap:5px;margin-bottom:12px;flex-wrap:wrap;align-items:center}
.filter-btn{padding:4px 12px;border-radius:16px;font-size:11px;font-weight:600;cursor:pointer;border:1px solid var(--border);background:transparent;color:var(--text-dim);transition:all .12s}
.filter-btn:hover{border-color:var(--accent);color:var(--accent-light)}
.filter-btn.active{background:var(--accent);border-color:var(--accent);color:#fff}
.filter-count{font-size:11px;color:var(--text-dim);margin-left:auto}

.ep-list{display:flex;flex-direction:column;gap:6px}
.ep-item{background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:12px;padding:12px 14px;transition:all .2s cubic-bezier(0.2,0.8,0.2,1)}
.ep-item:hover{border-color:rgba(255,255,255,.15);background:var(--card-hover);transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.2)}
.ep-item.disabled{opacity:.4}
.ep-item.current{border-color:var(--green);background:var(--green-dim)}
.ep-item.in-cooldown{border-color:var(--yellow);background:var(--yellow-dim)}
.ep-item.has-error{border-color:var(--red);background:var(--red-dim)}
.ep-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;gap:6px;flex-wrap:wrap}
.ep-name{font-weight:700;font-size:13px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.badge{font-size:9px;padding:2px 7px;border-radius:12px;font-weight:700;display:inline-flex;align-items:center;gap:3px;text-transform:uppercase;letter-spacing:.3px}
.badge-current{background:var(--green-dim);color:var(--green)}
.badge-disabled{background:var(--red-dim);color:var(--red)}
.badge-cooldown{background:var(--yellow-dim);color:var(--yellow)}
.badge-priority{background:var(--accent);color:#fff;min-width:20px;justify-content:center}
.badge-h-ok{background:var(--green-dim);color:var(--green)}
.badge-h-slow{background:var(--yellow-dim);color:var(--yellow)}
.badge-h-bad{background:var(--red-dim);color:var(--red)}
.badge-h-unknown{background:rgba(255,255,255,.06);color:var(--text-dim)}
.ep-meta{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:11px;color:var(--text-dim)}
.ep-meta span{display:flex;align-items:center;gap:3px}
.ep-actions{display:flex;gap:4px;flex-wrap:wrap}
.ep-error{margin-top:6px;font-size:11px;color:var(--red);background:var(--red-dim);padding:5px 8px;border-radius:6px;word-break:break-all}

.chain-list{display:flex;flex-direction:column;gap:0}
.chain-item{display:flex;align-items:center;gap:10px;padding:10px 12px;border-left:2px solid var(--border);font-size:12px;position:relative;transition:all .12s}
.chain-item:last-child{border-left-color:transparent}
.chain-item.active{border-left-color:var(--green);background:var(--green-dim);border-radius:0 8px 8px 0}
.chain-item.cooldown{border-left-color:var(--yellow);background:var(--yellow-dim);border-radius:0 8px 8px 0}
.chain-item.failed{border-left-color:var(--red);background:var(--red-dim);border-radius:0 8px 8px 0}
.chain-dot{width:8px;height:8px;border-radius:50%;background:var(--border);flex-shrink:0;position:absolute;left:-5px}
.chain-item.active .chain-dot{background:var(--green);box-shadow:0 0 6px var(--green)}
.chain-item.cooldown .chain-dot{background:var(--yellow)}
.chain-item.failed .chain-dot{background:var(--red)}
.chain-info{flex:1;min-width:0}
.chain-info .name{font-weight:600;font-size:12px}
.chain-info .model{font-size:10px;color:var(--text-dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chain-right{display:flex;flex-direction:column;align-items:flex-end;gap:2px;flex-shrink:0}
.chain-health{font-size:11px;font-weight:600}
.chain-err{font-size:10px;color:var(--red);max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chain-connector{height:12px;border-left:2px dashed var(--border);margin-left:0}

.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.65);backdrop-filter:blur(6px);display:flex;align-items:center;justify-content:center;z-index:1000;opacity:0;pointer-events:none;transition:opacity .15s}
.modal-overlay.show{opacity:1;pointer-events:all}
.modal{background:var(--card);backdrop-filter:var(--glass-blur);-webkit-backdrop-filter:var(--glass-blur);border:1px solid var(--border);border-radius:16px;padding:24px;width:560px;max-width:94vw;max-height:88vh;overflow-y:auto;box-shadow:0 16px 48px rgba(0,0,0,.5)}
.modal h2{font-size:16px;margin-bottom:18px;font-weight:700}
.form-group{margin-bottom:12px}
.form-group label{display:block;font-size:11px;font-weight:600;color:var(--text-dim);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
.form-group input,.form-group select{width:100%;padding:9px 11px;background:rgba(0,0,0,0.2);border:1px solid var(--border);border-radius:10px;color:var(--text);font-size:13px;outline:none;transition:border-color .12s;box-shadow:inset 0 1px 2px rgba(0,0,0,0.1)}
.form-group input:focus,.form-group select:focus{border-color:var(--accent)}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.form-actions{display:flex;gap:8px;justify-content:flex-end;margin-top:18px}

.model-row{display:flex;gap:8px;align-items:center}
.model-row input{flex:1}
.model-browser{margin-top:8px;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.mb-toolbar{display:flex;gap:6px;padding:8px 10px;background:rgba(255,255,255,.02);border-bottom:1px solid var(--border);align-items:center;flex-wrap:wrap}
.mb-toolbar input[type=text]{flex:1;min-width:100px;padding:6px 9px;background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:11px;outline:none}
.mb-toolbar input[type=text]:focus{border-color:var(--accent)}
.mb-toolbar label{font-size:11px;color:var(--text-dim);cursor:pointer;display:flex;align-items:center;gap:3px;white-space:nowrap}
.mb-toolbar .count{font-size:10px;color:var(--text-dim);white-space:nowrap}
.mb-table{max-height:300px;overflow-y:auto}
.mb-head{display:grid;grid-template-columns:28px 1fr 72px 80px 70px;gap:6px;padding:6px 10px;font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:.4px;font-weight:600;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--card);z-index:1}
.mb-row{display:grid;grid-template-columns:28px 1fr 72px 80px 70px;gap:6px;padding:6px 10px;align-items:center;border-bottom:1px solid rgba(255,255,255,.03);font-size:12px;cursor:pointer;transition:background .08s}
.mb-row:last-child{border-bottom:none}
.mb-row:hover{background:var(--card-hover)}
.mb-row.selected{background:var(--accent);color:#fff}
.mb-row input[type=checkbox]{accent-color:var(--accent);cursor:pointer}
.mb-row .name-cell{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mb-row .mm-cell{text-align:center;font-size:11px}
.mm-yes{color:var(--green)}
.mm-no{color:var(--text-dim)}
.mm-unknown{color:var(--text-dim);opacity:.5}
.mb-row .price-cell{font-size:10px;color:var(--text-dim);white-space:nowrap}
.mb-row.selected .price-cell{color:rgba(255,255,255,.6)}
.free-tag{font-size:9px;background:var(--green-dim);color:var(--green);padding:1px 5px;border-radius:8px}
.mb-row.selected .free-tag{background:rgba(255,255,255,.2);color:#fff}
.mb-row .lat-cell{font-size:10px;white-space:nowrap}
.lat-ok{color:var(--green)}
.lat-slow{color:var(--yellow)}
.lat-bad{color:var(--red)}
.batch-bar{display:flex;align-items:center;justify-content:space-between;padding:8px 10px;background:var(--accent);color:#fff;border-radius:7px;margin-top:8px;font-size:12px;font-weight:600}
.pagination{display:flex;align-items:center;justify-content:center;gap:4px;padding:8px;border-top:1px solid var(--border);background:rgba(255,255,255,.015)}
.pagination .btn{min-width:28px;justify-content:center}
.pagination .page-info{font-size:11px;color:var(--text-dim)}

.toast{position:fixed;bottom:20px;right:20px;padding:10px 16px;border-radius:8px;font-size:12px;font-weight:600;z-index:2000;opacity:0;transform:translateY(8px);transition:all .2s;max-width:320px;word-break:break-all}
.toast.show{opacity:1;transform:translateY(0)}
.toast-success{background:var(--green);color:#000}
.toast-error{background:var(--red);color:#fff}
.toast-info{background:var(--accent);color:#fff}

.empty{text-align:center;padding:32px 16px;color:var(--text-dim);font-size:13px}
.test-input-row{display:flex;gap:6px}
.test-input-row input{flex:1;padding:7px 10px;background:var(--bg);border:1px solid var(--border);border-radius:7px;color:var(--text);font-size:12px;outline:none}
.test-result{margin-top:8px;padding:8px 10px;border-radius:7px;font-size:11px;word-break:break-all;max-height:130px;overflow-y:auto;white-space:pre-wrap;font-family:'SF Mono',Menlo,Consolas,monospace}
.test-result.success{background:var(--green-dim);color:var(--green)}
.test-result.failure{background:var(--red-dim);color:var(--red)}

.log-card { background: var(--card); backdrop-filter: var(--glass-blur); -webkit-backdrop-filter: var(--glass-blur); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 18px; box-shadow: var(--shadow); display: flex; flex-direction: column; }
.log-container { height: 280px; overflow-y: auto; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 11px; display: flex; flex-direction: column; gap: 6px; scroll-behavior: smooth; }
.log-line { display: flex; gap: 8px; line-height: 1.5; word-break: break-all; }
.log-time { color: var(--text-dim); flex-shrink: 0; user-select: none; }
.log-INFO { color: var(--blue); flex-shrink: 0; min-width: 48px; text-align: center; }
.log-WARN { color: var(--yellow); flex-shrink: 0; min-width: 48px; text-align: center; }
.log-ERROR { color: var(--red); flex-shrink: 0; min-width: 48px; text-align: center; }
.log-msg { color: var(--text); }

.tabs { display:flex; background:rgba(255,255,255,0.03); border-radius:10px; padding:3px; border:1px solid var(--border); }
.tab { padding:6px 14px; border-radius:7px; font-size:12px; font-weight:600; cursor:pointer; color:var(--text-dim); transition:all .2s; }
.tab:hover { color:var(--text); }
.tab.active { background:var(--accent); color:#fff; box-shadow:0 2px 8px rgba(0,0,0,0.2); }

select option { background: var(--bg); color: var(--text); }

.seg-ctrl { display:inline-flex; background:rgba(255,255,255,0.03); border-radius:8px; padding:3px; border:1px solid rgba(255,255,255,0.05); }
.seg-btn { padding:3px 12px; border-radius:5px; font-size:11px; font-weight:600; cursor:pointer; color:var(--text-dim); transition:all 0.2s; }
.seg-btn:hover { color:var(--text); }
.seg-btn.active { background:rgba(255,255,255,0.1); color:#fff; box-shadow:0 2px 4px rgba(0,0,0,0.2); }

#testDrawer {
  position: fixed; right: 20px; bottom: 20px; width: 360px; background: rgba(20,20,20,0.85); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,0.6); z-index: 1000; display: flex; flex-direction: column; transform: translateY(150%); transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
#testDrawer.show { transform: translateY(0); }
.drawer-header { padding: 12px 16px; background: rgba(255,255,255,0.05); border-bottom: 1px solid var(--border); border-top-left-radius: 12px; border-top-right-radius: 12px; display: flex; justify-content: space-between; align-items: center; font-weight: bold; font-size: 13px; }
.drawer-body { padding: 16px; display: flex; flex-direction: column; gap: 10px; }
</style>

</head>
<body>

<div class="header">
  <div style="display:flex; align-items:center; gap:30px;">
      <h1><span class="logo">⚡</span> API Pool</h1>
      <div class="tabs">
          <div class="tab active" id="tabPool" onclick="switchTab('pool')">🔌 聚合池</div>
          <div class="tab" id="tabAnalytics" onclick="switchTab('analytics')">📊 数据面板</div>
      </div>
  </div>
  <div class="header-actions" id="poolActions">
    <button class="btn btn-ghost" onclick="runHealthCheck()">🩺 健康检测</button>
    <button class="btn btn-ghost" onclick="resetPool()">🔄 重置</button>
    <button class="btn btn-primary" onclick="openAddModal()">＋ 添加端点</button>
    <button class="btn btn-green" onclick="openTestDrawer('pool', '')">🧪 测试聚合池</button>
  </div>
  <div class="header-actions" id="analyticsActions" style="display:none;">
    <select id="analyticsFilter" class="btn btn-ghost" style="appearance:none; cursor:pointer; background:rgba(255,255,255,0.05);" onchange="loadAnalytics()">
        <option value="all">全端点统计</option>
    </select>
    <button class="btn btn-ghost" onclick="clearTokenStats()" style="color:var(--red);">🗑 清空统计</button>
    <button class="btn btn-green" onclick="exportCSV()">📥 导出流水</button>
  </div>
</div>

<div id="viewPool">
<div class="api-info-card">
  <div style="font-size: 13px; font-weight: 700; color: var(--accent-light); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;">🔗 客户端接入配置 (Client Config)</div>
  <div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px;">
    <div><span style="color: var(--text-dim); margin-right: 6px;">接口地址 (Base URL):</span><code id="displayUrl">http://localhost:5100/v1</code></div>
    <div><span style="color: var(--text-dim); margin-right: 6px;">API Key:</span><code>sk-any</code> <span style="font-size: 11px; color: var(--text-dim);">(任意填写)</span></div>
    <div><span style="color: var(--text-dim); margin-right: 6px;">模型 (Model):</span><code>api-pool</code> <span style="font-size: 11px; color: var(--text-dim);">(任意填写)</span></div>
    <div><span style="color: var(--text-dim); margin-right: 6px;">Responses API:</span><code>/v1/responses</code> <span style="font-size: 11px; color: var(--text-dim);">(入站适配)</span></div>
  </div>
</div>

<div class="grid">
  <div>
    <div class="stats" id="stats"></div>
    <div class="card" style="margin-bottom:16px">
      <div class="card-title"><span class="icon">📋</span> 端点列表</div>
      <div class="filter-bar" id="filterBar"></div>
      <div class="ep-list" id="epList"></div>
    </div>
  </div>
  <div>
    <div class="card" style="margin-bottom:16px">
      <div class="card-title"><span class="icon">🔗</span> 聚合链</div>
      <div style="font-size:10px;color:var(--text-dim);margin-bottom:10px">遇 429/超时自动切换 · 冷却到期自动切回</div>
      <div class="chain-list" id="chainList"></div>
    </div>

  </div>
</div>

<div class="log-card" style="margin-top:20px;">
  <div class="card-title">
    <span class="icon">📝</span> 实时日志
    <button class="btn btn-ghost btn-sm" onclick="clearSysLogs()" style="color:var(--red); float:right; margin-top:-2px; padding:2px 8px;">🗑 清空</button>
  </div>
  <div class="log-container" id="logContainer"></div>
</div>

<div class="log-card" style="margin-top:20px; display:flex; flex-direction:column; padding-bottom:15px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
      <div class="card-title" style="margin-bottom:0;"><span class="icon">💬</span> 对话日志 (Audit Logs)</div>
      <button class="btn btn-ghost btn-sm" onclick="clearChatLogs()" style="color:var(--red); padding:2px 8px;">🗑 清空记录</button>
    </div>
    <div style="display:flex; gap:15px; height:450px; min-height:300px;">
      <!-- List View -->
      <div style="flex:1; display:flex; flex-direction:column; border:1px solid var(--border); border-radius:8px; overflow:hidden;">
        <div style="background:rgba(255,255,255,0.05); padding:8px 12px; font-weight:bold; border-bottom:1px solid var(--border); display:grid; grid-template-columns: 80px 1fr 1fr 80px; gap:8px; font-size:12px;">
          <span>时间</span><span>端点</span><span>模型</span><span>Tokens</span>
        </div>
        <div id="chatLogsList" style="flex:1; overflow-y:auto;">
          <!-- Items inserted here -->
        </div>
        <div style="padding:8px; text-align:center; border-top:1px solid var(--border);">
            <button class="btn btn-ghost btn-sm" onclick="loadChatLogs(chatLogsPage-1)" id="clPrevBtn">上一页</button>
            <span style="margin:0 10px;font-size:12px;" id="clPageSpan">1</span>
            <button class="btn btn-ghost btn-sm" onclick="loadChatLogs(chatLogsPage+1)" id="clNextBtn">下一页</button>
        </div>
      </div>
      <!-- Detail View -->
      <div style="flex:1; display:flex; flex-direction:column; gap:15px; overflow:hidden; min-height:0;">
        <div style="flex:1; display:flex; flex-direction:column; border:1px solid var(--border); border-radius:8px; background:rgba(0,0,0,0.3); min-height:0;">
          <div style="padding:6px 10px; background:var(--card); font-size:12px; color:var(--text-dim); border-bottom:1px solid var(--border);">Prompt</div>
          <pre id="clPrompt" style="flex:1; overflow-y:auto; padding:10px; margin:0; font-size:12px; white-space:pre-wrap; word-break:break-all; min-height:0;"></pre>
        </div>
        <div style="flex:1; display:flex; flex-direction:column; border:1px solid var(--border); border-radius:8px; background:rgba(0,0,0,0.3); min-height:0;">
          <div style="padding:6px 10px; background:var(--card); font-size:12px; color:var(--text-dim); border-bottom:1px solid var(--border);">Completion <span id="clMeta" style="float:right;"></span></div>
          <pre id="clCompletion" style="flex:1; overflow-y:auto; padding:10px; margin:0; font-size:12px; white-space:pre-wrap; word-break:break-all; min-height:0;"></pre>
        </div>
      </div>
    </div>
</div>

</div>

<div id="viewAnalytics" style="display:none; padding-bottom:40px;">
    <div class="dash-stats" id="tokenStatsOverview"></div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0px;">
                <div class="card-title" style="margin-bottom:0; font-size:11px;">今日各时段消耗趋势</div>
                <div class="seg-ctrl">
                    <div class="seg-btn active" id="btnTrendToken" onclick="switchTrend('tokens')">Token</div>
                    <div class="seg-btn" id="btnTrendMissed" onclick="switchTrend('missed')">未命中</div>
                    <div class="seg-btn" id="btnTrendCall" onclick="switchTrend('calls')">请求数</div>
                </div>
            </div>
            <div id="tokenTodayChart" style="height: 180px; position: relative;"></div>
            <div id="missedTodayChart" style="height: 180px; position: relative; display:none;"></div>
            <div id="callsTodayChart" style="height: 180px; position: relative; display:none;"></div>
        </div>
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <div class="card-title" style="margin-bottom:0; font-size:11px;">近 14 天 Token 组成结构</div>
                <div style="display:flex; gap:10px; font-size:11px; color:var(--text-dim);">
                    <label style="cursor:pointer; display:flex; align-items:center; gap:3px;"><input type="checkbox" id="chkCompCache" checked onchange="updateCompChart()"> <span style="color:var(--green)">命中缓存</span></label>
                    <label style="cursor:pointer; display:flex; align-items:center; gap:3px;"><input type="checkbox" id="chkCompMissed" checked onchange="updateCompChart()"> <span style="color:var(--blue)">未命中</span></label>
                    <label style="cursor:pointer; display:flex; align-items:center; gap:3px;"><input type="checkbox" id="chkCompGen" checked onchange="updateCompChart()"> <span style="color:#aaa">生成</span></label>
                </div>
            </div>
            <div id="tokenTrendChart" style="height: 180px; margin-bottom: 0px; position: relative;"></div>
        </div>
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top:20px;">
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <div class="card-title" style="margin-bottom:0; font-size:11px;">今日模型端点排行榜</div>
                <div class="seg-ctrl">
                    <div class="seg-btn active" id="btnTblTodayToken" onclick="switchTblToday('tokens')">Token</div>
                    <div class="seg-btn" id="btnTblTodayCall" onclick="switchTblToday('calls')">请求数</div>
                </div>
            </div>
            <div style="max-height: 250px; overflow-y: auto;">
              <table style="width: 100%; border-collapse: collapse; font-size: 11px; text-align: left;">
                <thead><tr style="border-bottom: 1px solid var(--border); color: var(--text-dim);"><th style="padding: 6px;">模型端点</th><th style="padding: 6px; text-align:right;">数值</th></tr></thead>
                <tbody id="todayModelsTable"></tbody>
                <tbody id="todayCallsTable" style="display:none;"></tbody>
              </table>
            </div>
        </div>
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <div class="card-title" style="margin-bottom:0; font-size:11px;">本月模型端点排行榜</div>
                <div class="seg-ctrl">
                    <div class="seg-btn active" id="btnTblMonthToken" onclick="switchTblMonth('tokens')">Token</div>
                    <div class="seg-btn" id="btnTblMonthCall" onclick="switchTblMonth('calls')">请求数</div>
                </div>
            </div>
            <div style="max-height: 250px; overflow-y: auto;">
              <table style="width: 100%; border-collapse: collapse; font-size: 11px; text-align: left;">
                <thead><tr style="border-bottom: 1px solid var(--border); color: var(--text-dim);"><th style="padding: 6px;">模型端点</th><th style="padding: 6px; text-align:right;">数值</th></tr></thead>
                <tbody id="monthModelsTable"></tbody>
                <tbody id="monthCallsTable" style="display:none;"></tbody>
              </table>
            </div>
        </div>
    </div>
</div>

<div class="modal-overlay" id="modal">
  <div class="modal">
    <h2 id="modalTitle">添加端点</h2>
    <input type="hidden" id="editName">
    <div class="form-group"><label>名称</label><input type="text" id="fName" placeholder="如 OpenAI 或 DeepSeek" oninput="this.dataset.autofilled='false'"></div>
    <div class="form-group"><label>Base URL</label><input type="text" id="fUrl" placeholder="https://api.openai.com/v1" oninput="checkFetchBtn()"></div>
    <div class="form-group"><label>API Key</label><input type="password" id="fKey" placeholder="sk-..." oninput="checkFetchBtn()"></div>
    <div class="form-group">
      <label>模型</label>
      <div class="model-row">
        <input type="text" id="fModel" placeholder="gpt-4o">
        <button class="btn btn-yellow btn-sm" id="fetchModelsBtn" onclick="fetchModels()" disabled>🔍 获取</button>
      </div>
      <div id="modelBrowser" style="display:none"></div>
      <div id="batchBar" style="display:none"></div>
    </div>
    <div class="form-row">
      <div class="form-group"><label>优先级</label><input type="number" id="fPriority" value="1" min="1"></div>
      <div class="form-group"><label>超时 (秒)</label><input type="number" id="fTimeout" value="60" min="1"></div>
    </div>
    <div class="form-row">
      <div class="form-group"><label>重试次数</label><input type="number" id="fRetries" value="0" min="0"></div>
      <div class="form-group"><label>冷却 (分钟)</label><input type="number" id="fCooldown" value="5" min="0"></div>
    </div>
      <div class="form-group">
        <label title="标识该模型是否原生支持读图。若选“不支持”，收到图片时会自动触发图片解析。">多模态 (视觉) 能力</label>
        <select id="fVision"><option value="true">👁️ 原生支持</option><option value="false">🚫 不支持 (触发自动转译)</option></select>
      </div>
    <div class="form-row" style="grid-template-columns: 1fr 1fr;">
      <div class="form-group"><label title="该端点原生支持 xhigh 推理强度时开启，开启后不再把 xhigh 降级为 high">原生 xhigh 推理</label><select id="fXHigh"><option value="false">否 (自动降级)</option><option value="true">是 (原样透传)</option></select></div>
      <div class="form-group"><label title="该端点最高支持的推理档位，超过的请求会自动降到该档；留空表示不限制">最高推理档位</label><select id="fMaxEffort"><option value="">不限</option><option value="low">low</option><option value="medium">medium</option><option value="high">high</option><option value="xhigh">xhigh</option></select></div>
    </div>
    <div class="form-row">
      <div class="form-group"><label>启用</label><select id="fEnabled"><option value="true">是</option><option value="false">否</option></select></div>
      <div class="form-group"><label title="达到额度后挂起，0为不限制">每日额度 (0不限)</label><input type="number" id="fDailyLimit" value="0" min="0"></div>
    </div>
    <div class="form-row" style="grid-template-columns: 1fr 1fr 1fr;">
      <div class="form-group"><label title="每分钟最高请求次数，超限自动切换，0为不限制">并发 (0不限)</label><input type="number" id="fRpmLimit" value="0" min="0"></div>
      <div class="form-group"><label title="是否使用系统代理 (如v2ray)。本地或直连接口请选择否。">代理设置</label><select id="fProxy"><option value="true">随系统</option><option value="false">强制直连</option></select></div>
      <div class="form-group"><label title="底层协议类型">协议类型</label><select id="fProtocol"><option value="openai">OpenAI 兼容</option><option value="responses">OpenAI Responses</option><option value="anthropic">Anthropic</option></select></div>
    </div>
    <div class="form-group">
      <label>后台探针</label>
      <select id="fHealthMode">
        <option value="chat">Ping (/chat/completions)</option>
        <option value="models">Models (/v1/models) 零成本</option>
        <option value="none">关闭检测</option>
      </select>
    </div>
    <div class="form-actions">
      <button class="btn btn-ghost" onclick="closeModal()">取消</button>
      <button class="btn btn-green" id="batchAddBtn" style="display:none" onclick="batchAddEndpoints()">📦 批量添加</button>
      <button class="btn btn-primary" id="singleAddBtn" onclick="saveEndpoint()">保存</button>
    </div>
  </div>
</div>



<div class="toast" id="toast"></div>

<script>
document.getElementById('displayUrl').textContent = window.location.protocol + '//' + window.location.host + '/v1';

const API='';
async function api(method,path,body){const opts={method,headers:{'Content-Type':'application/json'}};if(body)opts.body=JSON.stringify(body);return(await fetch(API+path,opts)).json();}

let allModels=[],selectedModels=new Set(),latencyResults={},visionResults={};
let epFilter='all',modelPage=1,PP=50;

async function refresh(){
  const[eps,chain]=await Promise.all([api('GET','/api/endpoints'),api('GET','/api/chain')]);
  renderStats(eps);renderFilterBar(eps);renderEndpoints(eps);renderChain(chain);
}

function renderStats(eps){
  const t=eps.length,en=eps.filter(e=>e.enabled).length,h=eps.filter(e=>e.health==='ok').length;
  const cl=eps.filter(e=>e.in_cooldown).length,bad=eps.filter(e=>e.health==='bad').length;
  const calls=eps.reduce((s,e)=>s+e.total_calls,0);
  document.getElementById('stats').innerHTML=`
    <div class="stat-item"><div class="num">${t}</div><div class="label">端点</div></div>
    <div class="stat-item"><div class="num" style="color:var(--green)">${en}</div><div class="label">启用</div></div>
    <div class="stat-item"><div class="num" style="color:var(--green)">${h}</div><div class="label">健康</div></div>
    <div class="stat-item"><div class="num" style="color:${bad?'var(--red)':'var(--text-dim)'}">${bad}</div><div class="label">异常</div></div>
    <div class="stat-item"><div class="num" style="color:${cl?'var(--yellow)':'var(--text-dim)'}">${cl}</div><div class="label">冷却</div></div>
    <div class="stat-item"><div class="num" style="color:var(--blue)">${calls}</div><div class="label">调用</div></div>`;
}

function renderFilterBar(eps){
  const en=eps.filter(e=>e.enabled).length;
  document.getElementById('filterBar').innerHTML=`
    <button class="filter-btn ${epFilter==='all'?'active':''}" onclick="setFilter('all')">全部 ${eps.length}</button>
    <button class="filter-btn ${epFilter==='enabled'?'active':''}" onclick="setFilter('enabled')">启用 ${en}</button>
    <button class="filter-btn ${epFilter==='disabled'?'active':''}" onclick="setFilter('disabled')">禁用 ${eps.length-en}</button>
    <span class="filter-count" id="filterCount"></span>`;
}
function setFilter(f){epFilter=f;refresh();}

function hBadge(h,lat){
  const m={ok:['badge-h-ok','✅'],slow:['badge-h-slow','🐢'],bad:['badge-h-bad','❌'],unknown:['badge-h-unknown','❓'],testing:['badge-h-unknown','⏳']};
  const[c,l]=m[h]||m.unknown;
  return`<span class="badge ${c}">${l}${lat>=0?' '+lat+'ms':''}</span>`;
}

function renderEndpoints(eps){
  if(epFilter==='enabled')eps=eps.filter(e=>e.enabled);
  else if(epFilter==='disabled')eps=eps.filter(e=>!e.enabled);
  const c=document.getElementById('filterCount');if(c)c.textContent=`${eps.length} 个`;
  const el=document.getElementById('epList');
  if(!eps.length){el.innerHTML='<div class="empty">暂无端点</div>';return;}
  el.innerHTML=eps.map(ep=>{
    let cls='ep-item';
    if(!ep.enabled)cls+=' disabled';
    if(ep.is_current)cls+=' current';
    if(ep.in_cooldown)cls+=' in-cooldown';
    if(ep.last_error&&!ep.in_cooldown)cls+=' has-error';
    if(ep.daily_limit>0&&ep.today_used>=ep.daily_limit)cls+=' in-cooldown';
    if(ep.is_rpm_limited)cls+=' in-cooldown';
    let b=`<span class="badge badge-priority">#${ep.priority}</span>${hBadge(ep.health,ep.health_latency_ms)}`;
    if(ep.protocol==='anthropic')b+=`<span class="badge" style="background:rgba(217,119,87,0.2);color:#ff9e7a;border:1px solid rgba(217,119,87,0.3)" title="Anthropic 原生协议翻译">🧠Anthropic</span>`;
    else if(ep.protocol==='responses')b+=`<span class="badge" style="background:rgba(94,92,230,0.2);color:#9d9aff;border:1px solid rgba(94,92,230,0.3)" title="OpenAI Responses 原生协议">🔷Responses</span>`;
    else b+=`<span class="badge badge-priority" style="background:rgba(16,163,127,0.2);color:#2ecc71" title="OpenAI 兼容协议">🟢OpenAI</span>`;
    if(ep.is_current)b+='<span class="badge badge-current">● 当前</span>';
    if(!ep.enabled)b+='<span class="badge badge-disabled">禁用</span>';
      if(ep.is_vision!==false)b+=`<span class=\"badge\" style=\"background:rgba(0,122,255,.15);color:#0a84ff\" title=\"原生支持视觉能力\">👁️视觉</span>`;
    if(ep.supports_xhigh)b+=`<span class="badge" style="background:rgba(255,193,7,.15);color:#ffc107;border:1px solid rgba(255,193,7,.3)" title="原生支持 xhigh 推理强度">⚡xhigh</span>`;
    if(ep.max_reasoning_effort)b+=`<span class="badge" style="background:rgba(0,229,255,.12);color:#00d4ff;border:1px solid rgba(0,229,255,.28)" title="该端点最高支持到该推理档位">🔒max=${ep.max_reasoning_effort}</span>`;
    if(ep.is_rpm_limited)b+=`<span class="badge badge-cooldown" title="每分钟并发已满，限流降级中">🚧限流中</span>`;
    else if(ep.daily_limit>0&&ep.today_used>=ep.daily_limit)b+=`<span class="badge badge-cooldown" title="今日额度已满，挂起至明日">🛑额度耗尽</span>`;
    else if(ep.in_cooldown)b+=`<span class="badge badge-cooldown">⏳${fmtTime(ep.cooldown_remaining)}</span>`;
    if(!ep.use_proxy)b+=`<span class="badge badge-priority" title="绕过系统全局代理，强制直连">🌐直连</span>`;
    if(ep.health_mode==='none')b+=`<span class="badge" style="background:rgba(255,255,255,.05);color:var(--text-dim)" title="已关闭后台健康监测">🔕免扰</span>`;
    else if(ep.health_mode==='models')b+=`<span class="badge" style="background:rgba(50,215,75,.1);color:var(--green)" title="零成本 Models 探针">☘️无感测</span>`;
    if(ep.daily_limit>0)b+=`<span class="badge" style="background:rgba(255,255,255,.05);color:var(--text-dim)" title="每日消耗进度">📊${fmtNum(ep.today_used)} / ${fmtNum(ep.daily_limit)}</span>`;
    if(ep.rpm_limit>0)b+=`<span class="badge" style="background:rgba(255,255,255,.05);color:var(--text-dim)" title="每分钟最高并发请求限制">🚀${ep.rpm_limit} RPM</span>`;
    const last=ep.last_success?timeAgo(ep.last_success):'—';
    return`<div class="${cls}">
      <div class="ep-header">
        <div class="ep-name"><span style="word-break: break-all;">${esc(ep.name)}</span> ${b}</div>
        <div class="ep-actions">
          <button class="btn btn-ghost btn-sm" title="连通性测试" onclick="openTestDrawer('${ep.id}', '${esc(ep.name)} (${esc(ep.model)})')">🧪</button>
          ${ep.in_cooldown?`<button class="btn btn-yellow btn-sm" title="立刻解除冷却" onclick="clearCooldown('${ep.id}')">⏰</button>`:''}
          <button class="btn btn-ghost btn-sm" title="${ep.enabled?'禁用端点':'启用端点'}" onclick="toggleEndpoint('${ep.id}')">${ep.enabled?'⏸':'▶'}</button>
          <button class="btn btn-ghost btn-sm" title="编辑端点" onclick="editEndpoint('${ep.id}')">✏️</button>
          <button class="btn btn-ghost btn-sm" title="删除端点" onclick="deleteEndpoint('${ep.id}', '${esc(ep.name)}')" style="color:var(--red)">🗑</button>
        </div>
      </div>
      <div class="ep-meta">
        <span title="绑定的模型名称">🤖${esc(ep.model)}</span><span title="单次请求超时时间">⏱${ep.timeout}s</span><span title="失败后最大重试次数">🔁${ep.max_retries}次</span><span title="请求失败后的冷却惩罚时间">❄️${ep.cooldown_minutes}分</span><span title="累计成功响应次数">📞${ep.total_calls}次</span><span title="最后一次成功响应时间">🕐${last}</span>
      </div>
      ${ep.last_error?`<div class="ep-error">⚠ ${esc(ep.last_error)}</div>`:''}
    </div>`;
  }).join('');
}

function renderChain(chain){
  const el=document.getElementById('chainList');
  if(!chain.length){el.innerHTML='<div class="empty">没有启用的端点</div>';return;}
  el.innerHTML=chain.map((it,i)=>{
    let cls='chain-item';
    if(it.in_cooldown)cls+=' cooldown';
    else if(it.is_current)cls+=' active';
    else if(it.health==='bad'||it.fail_count>0)cls+=' failed';
    const st=it.in_cooldown?'<span class="badge badge-warning">冷却中</span>':(it.is_current?'<span class="badge badge-success">服务中</span>':'');
    const h=it.health, lat=it.health_latency_ms;
    let rh='';
    if(h==='ok')rh=`<div class="chain-health" style="color:var(--green)">✓${lat>=0?' '+lat+'ms':''}</div>`;
    else if(h==='slow')rh=`<div class="chain-health" style="color:var(--yellow)">🐢${lat>=0?' '+lat+'ms':''}</div>`;
    else if(h==='bad'){rh=`<div class="chain-health" style="color:var(--red)">✗${lat>=0?' '+lat+'ms':''}</div>`;if(it.health_error)rh+=`<div class="chain-err" title="${esc(it.health_error)}">${esc(it.health_error)}</div>`;}
    else if(h==='testing')rh='<div class="chain-health" style="color:var(--text-dim)">…</div>';
    else rh='<div class="chain-health" style="color:var(--text-dim)">-</div>';
    const conn=i<chain.length-1?'<div class="chain-connector"></div>':'';
    const vis = (it.is_vision !== false) ? '<span class="badge" style="background:rgba(0,122,255,.15);color:#0a84ff" title="支持原生视觉">👁️视觉</span>' : '';
    return`<div class="${cls}"><div class="chain-dot"></div><div class="chain-info"><div class="name">${esc(it.name)} ${st}</div><div class="model">${esc(it.model)} ${vis}</div></div><div class="chain-right">${rh}</div></div>${conn}`;
  }).join('');
}

async function runHealthCheck(){toast('正在检测...','info');const r=await api('POST','/api/health-check');if(r.ok){const o=r.results.filter(x=>x.health==='ok').length,s=r.results.filter(x=>x.health==='slow').length,b=r.results.filter(x=>x.health==='bad').length;toast(`✅${o} 🐢${s} ❌${b}`,'success');}refresh();}
async function toggleEndpoint(id){await api('POST',`/api/endpoints/${encodeURIComponent(id)}/toggle`);refresh();}
async function deleteEndpoint(id, n){if(!confirm(`删除「${n}」？`))return;await api('DELETE',`/api/endpoints/${encodeURIComponent(id)}`);toast('已删除','success');refresh();}
async function clearCooldown(id){await api('PUT',`/api/endpoints/${encodeURIComponent(id)}`,{cooldown_minutes:0});await api('POST','/api/reset');setTimeout(async()=>{await api('PUT',`/api/endpoints/${encodeURIComponent(id)}`,{cooldown_minutes:5});refresh();},200);toast('已解除冷却','success');refresh();}
let testImageBase64 = null;
function previewTestImage(input) {
  if (input.files && input.files[0]) {
    const reader = new FileReader();
    reader.onload = function(e) {
      testImageBase64 = e.target.result;
      const btn = document.getElementById('btnTestImage');
      if(btn) { btn.style.background = 'rgba(50,215,75,0.2)'; btn.title = '已附加图片'; }
      toast('图片已附加', 'success');
    };
    reader.readAsDataURL(input.files[0]);
  }
}
function clearTestImage() {
  testImageBase64 = null;
  const input = document.getElementById('testImage');
  if (input) input.value = '';
  const btn = document.getElementById('btnTestImage');
  if (btn) { btn.style.background = 'transparent'; btn.title = '上传图片测试'; }
}

function openTestDrawer(targetId, targetName) {
  document.getElementById('testTargetId').value = targetId;
  if(targetId === 'pool') {
    document.getElementById('testDrawerTitle').innerHTML = '🧪 测试端点池 (Pool)';
  } else {
    document.getElementById('testDrawerTitle').innerHTML = `🧪 测试: ${targetName}`;
  }
  document.getElementById('testResult').style.display = 'none';
  document.getElementById('testDrawer').classList.add('show');
}
function closeTestDrawer() {
  document.getElementById('testDrawer').classList.remove('show');
}
async function sendTest() {
  const targetId = document.getElementById('testTargetId').value;
  const m = document.getElementById('testMsg').value || '你好';
  toast('发送测试中...','info');
  let r;
  if(targetId === 'pool'){
    r = await api('POST','/api/test-pool',{message:m,image:testImageBase64});
  } else {
    r = await api('POST','/api/test',{id:targetId,message:m,image:testImageBase64});
  }
  const el = document.getElementById('testResult');
  el.style.display = 'block';
  if(r.ok){
    el.className='test-result success';
    el.textContent='✅ '+r.result+(r.served_by?'\n[响应: '+r.served_by+']':'');
  }else{
    el.className='test-result failure';
    el.textContent='❌ '+(r.error||r.errors?.join('\n'));
  }
  refresh();
}
async function resetPool(){await api('POST','/api/reset');toast('已重置','success');refresh();}

function checkFetchBtn(){
    const u=document.getElementById('fUrl').value.trim(),k=document.getElementById('fKey').value.trim();
    document.getElementById('fetchModelsBtn').disabled=!(u&&k);
    
    const nameEl = document.getElementById('fName');
    if (!nameEl.value || nameEl.dataset.autofilled === 'true') {
        const provider = detectProvider(u);
        if (provider) {
            nameEl.value = provider;
            nameEl.dataset.autofilled = 'true';
        } else if (nameEl.dataset.autofilled === 'true') {
            nameEl.value = '';
            nameEl.dataset.autofilled = 'false';
        }
    }
}
function detectProvider(url) {
    if(!url) return '';
    const u = url.toLowerCase();
    if(u.includes('api.openai.com')) return 'OpenAI';
    if(u.includes('openrouter.ai')) return 'OpenRouter';
    if(u.includes('api.anthropic.com')) return 'Anthropic';
    if(u.includes('api.deepseek.com')) return 'DeepSeek';
    if(u.includes('integrate.api.nvidia.com')) return 'NVIDIA';
    if(u.includes('open.bigmodel.cn')) return 'BigModel';
    if(u.includes('dashscope.aliyuncs.com')) return 'Aliyun';
    if(u.includes('api.siliconflow.cn')) return 'SiliconFlow';
    if(u.includes('api.moonshot.cn')) return 'Moonshot';
    if(u.includes('api.groq.com')) return 'Groq';
    if(u.includes('api.together.xyz')) return 'Together';
    if(u.includes('ollama')) return 'Ollama';
    if(u.includes('localhost') || u.includes('127.0.0.1')) return 'Local';
    try {
        const dom = new URL(url).hostname;
        const parts = dom.split('.');
        if(parts.length >= 2) {
            let name = parts[parts.length-2];
            return name.charAt(0).toUpperCase() + name.slice(1);
        }
    }catch(e){}
    return '';
}
async function fetchModels(){
  const u=document.getElementById('fUrl').value.trim(),k=document.getElementById('fKey').value.trim();
  const up=document.getElementById('fProxy').value==='true',pt=document.getElementById('fProtocol').value||'openai';
  if(!u||!k){toast('填写 URL 和 Key','error');return;}
  const b=document.getElementById('fetchModelsBtn');b.disabled=true;b.innerHTML='⏳';
  try{const r=await api('POST','/api/fetch-models',{base_url:u,api_key:k,use_proxy:up,protocol:pt});
    if(r.ok&&r.models?.length){allModels=r.models;selectedModels=new Set();latencyResults={};visionResults={};modelPage=1;renderModelBrowser();toast(`${r.count} 个模型`,'success');}
    else{document.getElementById('modelBrowser').innerHTML=`<div style="padding:10px;color:var(--red);font-size:12px">❌ ${esc(r.error||'无模型')}</div>`;document.getElementById('modelBrowser').style.display='block';}
  }catch(e){toast('请求失败','error');}
  b.disabled=false;b.innerHTML='🔍 获取';
}
function isOpenRouter(){return document.getElementById('fUrl').value.includes('openrouter');}
function isFreeModel(m){if(!m.pricing)return false;return parseFloat(m.pricing.prompt||'1')===0&&parseFloat(m.pricing.completion||'1')===0;}

function renderModelBrowser(){
  const el=document.getElementById('modelBrowser');
  el.innerHTML=`<div class="model-browser">
    <div class="mb-toolbar">
      <input type="text" id="modelSearch" placeholder="搜索..." oninput="modelPage=1;filterModels()">
      <button class="btn btn-ghost btn-sm" onclick="selectAll()">全选</button>
      <button class="btn btn-ghost btn-sm" onclick="selectNone()">清空</button>
      <button class="btn btn-ghost btn-sm" onclick="testSelectedLatency()">⏱ 延迟</button>
      <button class="btn btn-ghost btn-sm" onclick="testSelectedVision()">🖼 多模态</button>
      ${isOpenRouter()?`<label><input type="checkbox" id="freeOnly" onchange="modelPage=1;filterModels()"> 🆓免费</label>`:''}
      <span class="count" id="modelCount"></span>
    </div>
    <div class="mb-head"><span></span><span>模型</span><span style="text-align:center">多模态</span><span>价格</span><span>延迟</span></div>
    <div class="mb-table" id="modelListInner"></div>
    <div class="pagination" id="modelPagination" style="display:none"></div>
  </div>`;
  el.style.display='block';
  filterModels();
}

function getFilteredModels(){
  const q=(document.getElementById('modelSearch')?.value||'').toLowerCase();
  const fo=document.getElementById('freeOnly')?.checked||false;
  let f=allModels;if(fo)f=f.filter(m=>isFreeModel(m));if(q)f=f.filter(m=>m.id.toLowerCase().includes(q));return f;
}

function filterModels(){
  const f=getFilteredModels();
  const tp=Math.max(1,Math.ceil(f.length/PP));
  if(modelPage>tp)modelPage=tp;
  const si=(modelPage-1)*PP,pg=f.slice(si,si+PP);
  const inner=document.getElementById('modelListInner');
  const c=document.getElementById('modelCount');if(c)c.textContent=`${f.length}/${allModels.length}`;
  inner.innerHTML=pg.map(m=>{
    const sel=selectedModels.has(m.id);
    const vr=visionResults[m.id];
    let mm='<span class="mm-unknown">—</span>';
    if(vr){mm=vr.supports_vision?'<span class="mm-yes">✅ 图片</span>':'<span class="mm-no">❌</span>';}
    const lat=latencyResults[m.id];
    let lh='';
    if(lat){if(lat.status==='ok')lh=`<span class="lat-ok">✓${lat.latency_ms}ms</span>`;else if(lat.status==='slow')lh=`<span class="lat-slow">🐢${lat.latency_ms}ms</span>`;else lh=`<span class="lat-bad">✗${lat.latency_ms}ms</span>`;}
    let ph='';
    if(m.pricing){if(isFreeModel(m))ph='<span class="free-tag">FREE</span>';else ph=`<span style="font-size:10px;color:var(--text-dim)">$${m.pricing.prompt||'0'}/$${m.pricing.completion||'0'}</span>`;}
    return`<div class="mb-row${sel?' selected':''}" onclick="event.target.tagName!=='INPUT'&&toggleModel('${esc(m.id)}')">
      <input type="checkbox" ${sel?'checked':''} onclick="event.stopPropagation();toggleModel('${esc(m.id)}')">
      <span class="name-cell" title="${esc(m.id)}">${esc(m.id)}</span>
      <span class="mm-cell">${mm}</span>
      <span class="price-cell">${ph}</span>
      <span class="lat-cell">${lh}</span>
    </div>`;
  }).join('');
  const pag=document.getElementById('modelPagination');
  if(tp>1){pag.style.display='flex';pag.innerHTML=`
    <button class="btn btn-ghost btn-sm" onclick="modelPage=1;filterModels()" ${modelPage===1?'disabled':''}>⏮</button>
    <button class="btn btn-ghost btn-sm" onclick="modelPage--;filterModels()" ${modelPage===1?'disabled':''}>◀</button>
    <span class="page-info">${modelPage}/${tp}</span>
    <button class="btn btn-ghost btn-sm" onclick="modelPage++;filterModels()" ${modelPage===tp?'disabled':''}>▶</button>
    <button class="btn btn-ghost btn-sm" onclick="modelPage=${tp};filterModels()" ${modelPage===tp?'disabled':''}>⏭</button>`;}
  else pag.style.display='none';
  updateBatchBar();
}

function toggleModel(id){selectedModels.has(id)?selectedModels.delete(id):selectedModels.add(id);if(selectedModels.size===1)document.getElementById('fModel').value=[...selectedModels][0];else if(!selectedModels.size)document.getElementById('fModel').value='';filterModels();}
function selectAll(){getFilteredModels().forEach(m=>selectedModels.add(m.id));filterModels();}
function selectNone(){selectedModels.clear();document.getElementById('fModel').value='';filterModels();}

function updateBatchBar(){
  const bar=document.getElementById('batchBar'),bb=document.getElementById('batchAddBtn'),sb=document.getElementById('singleAddBtn');
  if(selectedModels.size>1){bar.style.display='block';bar.innerHTML=`<div class="batch-bar"><span>已选 ${selectedModels.size} 个模型</span></div>`;bb.style.display='inline-flex';sb.style.display='none';}
  else{bar.style.display='none';bb.style.display='none';sb.style.display='inline-flex';}
}

async function testSelectedLatency(){
  const u=document.getElementById('fUrl').value.trim(),k=document.getElementById('fKey').value.trim();
  const up=document.getElementById('fProxy').value==='true',pt=document.getElementById('fProtocol').value||'openai';
  if(!u||!k){toast('填写 URL 和 Key','error');return;}
  if(!selectedModels.size){toast('勾选模型','error');return;}
  const ms=[...selectedModels];toast(`测试 ${ms.length} 个...`,'info');
  for(const mid of ms){latencyResults[mid]={status:'bad',latency_ms:0};filterModels();try{latencyResults[mid]=await api('POST','/api/test-model',{base_url:u,api_key:k,model:mid,use_proxy:up,protocol:pt});}catch(e){latencyResults[mid]={status:'bad',latency_ms:0};}filterModels();}
  toast(`✅${Object.values(latencyResults).filter(r=>r.ok).length}/${ms.length}`,'success');
}

async function testSelectedVision(){
  const u=document.getElementById('fUrl').value.trim(),k=document.getElementById('fKey').value.trim();
  const up=document.getElementById('fProxy').value==='true',pt=document.getElementById('fProtocol').value||'openai';
  if(!u||!k){toast('填写 URL 和 Key','error');return;}
  if(!selectedModels.size){toast('勾选模型','error');return;}
  const ms=[...selectedModels];toast(`检测 ${ms.length} 个多模态...`,'info');
  let vis=0;
  for(const mid of ms){visionResults[mid]={supports_vision:false};filterModels();try{const r=await api('POST','/api/test-vision',{base_url:u,api_key:k,model:mid,use_proxy:up,protocol:pt});visionResults[mid]=r;if(r.supports_vision)vis++;}catch(e){visionResults[mid]={supports_vision:false};}filterModels();}
  toast(`多模态: ${vis}/${ms.length} 支持`,'success');
}

function openAddModal(){
    document.getElementById('editName').value='';document.getElementById('modalTitle').textContent='添加端点';
    ['fName','fUrl','fKey','fModel'].forEach(id=>document.getElementById(id).value='');
    document.getElementById('fPriority').value=1;document.getElementById('fTimeout').value=60;document.getElementById('fRetries').value=0;document.getElementById('fCooldown').value=5;document.getElementById('fEnabled').value='true';document.getElementById('fDailyLimit').value=0;document.getElementById('fRpmLimit').value=0;document.getElementById('fProxy').value='true';document.getElementById('fProtocol').value='openai';document.getElementById('fHealthMode').value='chat';document.getElementById('fVision').value='true';document.getElementById('fXHigh').value='false';document.getElementById('fMaxEffort').value='';
    document.getElementById('modelBrowser').style.display='none';document.getElementById('batchBar').style.display='none';
    document.getElementById('fetchModelsBtn').disabled=true;document.getElementById('batchAddBtn').style.display='none';document.getElementById('singleAddBtn').style.display='inline-flex';
    allModels=[];selectedModels=new Set();latencyResults={};visionResults={};
    document.getElementById('modal').classList.add('show');
}
function editEndpoint(id){
    api('GET','/api/endpoints').then(eps=>{const ep=eps.find(e=>e.id===id);if(!ep)return;
        document.getElementById('editName').value=id;document.getElementById('modalTitle').textContent='编辑端点';
        document.getElementById('fName').value=ep.name;document.getElementById('fUrl').value=ep.base_url;document.getElementById('fKey').value=ep.api_key_full||'';document.getElementById('fModel').value=ep.model;
        document.getElementById('fPriority').value=ep.priority;document.getElementById('fTimeout').value=ep.timeout;document.getElementById('fRetries').value=ep.max_retries;document.getElementById('fCooldown').value=ep.cooldown_minutes;document.getElementById('fEnabled').value=String(ep.enabled);document.getElementById('fDailyLimit').value=ep.daily_limit||0;document.getElementById('fRpmLimit').value=ep.rpm_limit||0;document.getElementById('fProxy').value=String(ep.use_proxy!==false);document.getElementById('fProtocol').value=ep.protocol||'openai';document.getElementById('fHealthMode').value=ep.health_mode||'chat';document.getElementById('fVision').value=String(ep.is_vision!==false);document.getElementById('fXHigh').value=String(!!ep.supports_xhigh);document.getElementById('fMaxEffort').value=ep.max_reasoning_effort||'';
        document.getElementById('modelBrowser').style.display='none';document.getElementById('batchBar').style.display='none';document.getElementById('batchAddBtn').style.display='none';document.getElementById('singleAddBtn').style.display='inline-flex';
        allModels=[];selectedModels=new Set();latencyResults={};visionResults={};checkFetchBtn();document.getElementById('modal').classList.add('show');
    });
}
function closeModal(){document.getElementById('modal').classList.remove('show');}

async function saveEndpoint(){
    const ep_id=document.getElementById('editName').value;
    const d={name:document.getElementById('fName').value.trim(),base_url:document.getElementById('fUrl').value.trim(),api_key:document.getElementById('fKey').value.trim(),model:document.getElementById('fModel').value.trim(),priority:parseInt(document.getElementById('fPriority').value)||1,timeout:parseInt(document.getElementById('fTimeout').value)||60,max_retries:parseInt(document.getElementById('fRetries').value)||0,cooldown_minutes:parseInt(document.getElementById('fCooldown').value)||0,enabled:document.getElementById('fEnabled').value==='true',daily_limit:parseInt(document.getElementById('fDailyLimit').value)||0,rpm_limit:parseInt(document.getElementById('fRpmLimit').value)||0,use_proxy:document.getElementById('fProxy').value==='true',protocol:document.getElementById('fProtocol').value||'openai',health_mode:document.getElementById('fHealthMode').value||'chat',is_vision:document.getElementById('fVision').value==='true',supports_xhigh:document.getElementById('fXHigh').value==='true',max_reasoning_effort:document.getElementById('fMaxEffort').value};
    if(!d.name||!d.base_url||!d.api_key){toast('填写名称/URL/Key','error');return;}
    if(!d.model){toast('选择模型','error');return;}
    if(ep_id){await api('PUT',`/api/endpoints/${encodeURIComponent(ep_id)}`,d);toast('已更新','success');}
    else{await api('POST','/api/endpoints',d);toast('已添加','success');}
    closeModal();refresh();
}

async function batchAddEndpoints(){
    const fn=document.getElementById('fName').value.trim();
    const u=document.getElementById('fUrl').value.trim(),k=document.getElementById('fKey').value.trim();
    const sp=parseInt(document.getElementById('fPriority').value)||1,to=parseInt(document.getElementById('fTimeout').value)||60,re=parseInt(document.getElementById('fRetries').value)||0,cd=parseInt(document.getElementById('fCooldown').value)||5,dl=parseInt(document.getElementById('fDailyLimit').value)||0,rl=parseInt(document.getElementById('fRpmLimit').value)||0,up=document.getElementById('fProxy').value==='true',pt=document.getElementById('fProtocol').value||'openai',hm=document.getElementById('fHealthMode').value||'chat';
    if(!u||!k){toast('填写 URL 和 Key','error');return;}
    if(!selectedModels.size){toast('选择模型','error');return;}
    const ms=[...selectedModels];toast(`添加 ${ms.length} 个...`,'info');
    const r=await api('POST','/api/endpoints/batch',{endpoints:ms.map((m,i)=>({name:fn?fn:m,model:m,priority:sp+i,is_vision:visionResults[m]?visionResults[m].supports_vision:true})),base:{base_url:u,api_key:k,timeout:to,max_retries:re,cooldown_minutes:cd,daily_limit:dl,rpm_limit:rl,use_proxy:up,protocol:pt,start_priority:sp,health_mode:hm}});
    if(r.ok){toast(`✅ ${r.added} 个`,'success');closeModal();refresh();}else toast('失败','error');
}

function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function timeAgo(ts){if(!ts)return'—';const s=Math.floor(Date.now()/1000-ts);if(s<60)return s+'s前';if(s<3600)return Math.floor(s/60)+'m前';if(s<86400)return Math.floor(s/3600)+'h前';return Math.floor(s/86400)+'d前';}
function fmtTime(s){if(s<=0)return'';if(s<60)return s+'s';const m=Math.floor(s/60);return(s%60)?`${m}m${s%60}s`:`${m}m`;}
function toast(msg,type){const el=document.getElementById('toast');el.textContent=msg;el.className='toast toast-'+type+' show';setTimeout(()=>el.classList.remove('show'),2500);}

function closeStatsModal(){document.getElementById('statsModal').classList.remove('show');}
function fmtNum(n) {
    if (!n) return '0';
    if (n >= 100000000) return (n / 100000000).toFixed(2).replace(/\.00$/, '') + ' 亿';
    if (n >= 10000) return (n / 10000).toFixed(2).replace(/\.00$/, '') + ' 万';
    return n.toLocaleString();
}

function drawSVGChart(containerId, data, options = {}) {
    const key = options.key || 'tokens';
    const unit = options.unit || 'Tokens';
    const container = document.getElementById(containerId);
    if (!data || data.length === 0) {
        if(container) container.innerHTML = '<div class="empty">暂无趋势数据</div>';
        return;
    }
    const maxVal = Math.max(...data.map(d => d[key])) || 1;
    const padding = 15;
    const w = container.clientWidth || 800;
    const h = 180;
    
    let pts = data.map((d, i) => {
        const x = padding + (i / Math.max(1, data.length - 1)) * (w - 2 * padding);
        const y = h - padding - (d[key] / maxVal) * (h - 2 * padding);
        return {x, y, d};
    });
    
    let pathD = pts.length ? `M ${pts[0].x},${pts[0].y}` : '';
    for (let i = 1; i < pts.length; i++) {
        const prev = pts[i-1], curr = pts[i];
        const cpX = prev.x + (curr.x - prev.x) / 2;
        pathD += ` C ${cpX},${prev.y} ${cpX},${curr.y} ${curr.x},${curr.y}`;
    }
    const polyD = pts.length ? `${pathD} L ${pts[pts.length-1].x},${h} L ${pts[0].x},${h} Z` : '';
    
    const yTicks = [maxVal, maxVal/2, 0];
    const yTickElements = yTicks.map(val => `<text x="5" y="${h - padding - (val/maxVal)*(h - 2*padding) - 4}" fill="var(--text-dim)" font-size="10" font-family="monospace">${fmtNum(val)}</text>`).join('');

    container.innerHTML = `
        <svg viewBox="0 0 ${w} ${h}" style="width:100%; height:100%; overflow:visible;">
            <defs>
                <linearGradient id="chartGrad_${containerId}" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="rgba(124, 109, 240, 0.5)"/>
                    <stop offset="100%" stop-color="rgba(124, 109, 240, 0.0)"/>
                </linearGradient>
            </defs>
            <line x1="0" y1="${padding}" x2="${w}" y2="${padding}" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4"/>
            <line x1="0" y1="${h/2}" x2="${w}" y2="${h/2}" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4"/>
            <line x1="0" y1="${h-padding}" x2="${w}" y2="${h-padding}" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4"/>
            ${yTickElements}
            <path d="${polyD}" fill="url(#chartGrad_${containerId})"/>
            <path d="${pathD}" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round"/>
            ${pts.map((p, i) => `<circle cx="${p.x}" cy="${p.y}" r="4" fill="var(--bg)" stroke="var(--accent)" stroke-width="2" class="chart-point" data-idx="${i}" style="cursor:pointer; transition:all 0.2s;"/>`).join('')}
        </svg>
        <div id="${containerId}_tt" style="position:absolute; display:none; background:rgba(20,20,25,0.95); backdrop-filter:blur(10px); border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:8px 12px; font-size:11px; box-shadow:0 8px 32px rgba(0,0,0,0.5); pointer-events:none; z-index:100; white-space:nowrap; transition: left 0.1s, top 0.1s;"></div>
    `;
    
    const tt = document.getElementById(`${containerId}_tt`);
    container.querySelectorAll('.chart-point').forEach(c => {
        c.addEventListener('mouseenter', (e) => {
            const idx = e.target.getAttribute('data-idx');
            const d = data[idx];
            c.setAttribute('r', '6');
            c.style.filter = 'drop-shadow(0 0 4px var(--accent-light))';
            tt.style.display = 'block';
            tt.innerHTML = `<div style="color:var(--text-dim);margin-bottom:4px;font-weight:600">${d.date}</div><div style="font-weight:700;color:var(--accent-light);font-size:13px;">${fmtNum(d[key])} <span style="font-size:10px;font-weight:400;color:var(--text-dim)">${unit}</span></div>`;
            
            let tx = parseFloat(c.getAttribute('cx')) + 12;
            let ty = parseFloat(c.getAttribute('cy')) - 35;
            if (tx + 120 > container.clientWidth) tx = container.clientWidth - 130;
            if (ty < 0) ty = parseFloat(c.getAttribute('cy')) + 15;
            if (ty + 50 > container.clientHeight) ty = container.clientHeight - 55;
            tt.style.left = tx + 'px';
            tt.style.top = ty + 'px';
        });
        c.addEventListener('mouseleave', () => {
            c.setAttribute('r', '4');
            c.style.filter = 'none';
            tt.style.display = 'none';
        });
    });
}

function drawCompositionChart(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container || !data || !data.length) {
        if(container) container.innerHTML = '<div class="empty">暂无数据</div>';
        return;
    }
    const showCache = document.getElementById('chkCompCache')?.checked ?? true;
    const showMissed = document.getElementById('chkCompMissed')?.checked ?? true;
    const showGen = document.getElementById('chkCompGen')?.checked ?? true;

    const maxVal = Math.max(...data.map(d => {
        const c1 = showCache ? (d.cached || 0) : 0;
        const c2 = c1 + (showMissed ? Math.max(0, (d.prompt || 0) - (d.cached || 0)) : 0);
        return c2 + (showGen ? Math.max(0, (d.tokens || 0) - (d.prompt || 0)) : 0);
    })) || 1;
    const padding = 15;
    const w = container.clientWidth || 800;
    const h = 180;
    
    let pts1 = [], pts2 = [], pts3 = [];
    data.forEach((d, i) => {
        const x = padding + (i / Math.max(1, data.length - 1)) * (w - 2 * padding);
        const c1 = showCache ? (d.cached || 0) : 0;
        const c2 = c1 + (showMissed ? Math.max(0, (d.prompt || 0) - (d.cached || 0)) : 0);
        const c3 = c2 + (showGen ? Math.max(0, (d.tokens || 0) - (d.prompt || 0)) : 0);
        
        if(showCache) pts1.push({x, y: h - padding - (c1 / maxVal) * (h - 2 * padding)});
        if(showMissed) pts2.push({x, y: h - padding - (c2 / maxVal) * (h - 2 * padding)});
        if(showGen) pts3.push({x, y: h - padding - (c3 / maxVal) * (h - 2 * padding)});
    });
    
    const genPath = (pts) => {
        if (!pts.length) return '';
        let dStr = `M ${pts[0].x},${pts[0].y}`;
        for (let i = 1; i < pts.length; i++) {
            const prev = pts[i-1], curr = pts[i];
            const cpX = prev.x + (curr.x - prev.x) / 2;
            dStr += ` C ${cpX},${prev.y} ${cpX},${curr.y} ${curr.x},${curr.y}`;
        }
        return dStr;
    };
    
    const path1 = genPath(pts1), path2 = genPath(pts2), path3 = genPath(pts3);
    const poly1 = pts1.length ? `${path1} L ${pts1[pts1.length-1].x},${h} L ${pts1[0].x},${h} Z` : '';
    const poly2 = pts2.length ? `${path2} L ${pts2[pts2.length-1].x},${h} L ${pts2[0].x},${h} Z` : '';
    const poly3 = pts3.length ? `${path3} L ${pts3[pts3.length-1].x},${h} L ${pts3[0].x},${h} Z` : '';
    
    const yTicks = [maxVal, maxVal/2, 0];
    const yTickElements = yTicks.map(val => `<text x="5" y="${h - padding - (val/maxVal)*(h - 2*padding) - 4}" fill="var(--text-dim)" font-size="10" font-family="monospace">${fmtNum(val)}</text>`).join('');

    container.innerHTML = `
        <svg viewBox="0 0 ${w} ${h}" style="width:100%; height:100%; overflow:visible;">
            <defs>
                <linearGradient id="g3" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="rgba(150, 150, 150, 0.4)"/><stop offset="100%" stop-color="rgba(150, 150, 150, 0.0)"/></linearGradient>
                <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="rgba(80, 150, 255, 0.5)"/><stop offset="100%" stop-color="rgba(80, 150, 255, 0.0)"/></linearGradient>
                <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="rgba(0, 200, 100, 0.5)"/><stop offset="100%" stop-color="rgba(0, 200, 100, 0.0)"/></linearGradient>
            </defs>
            <line x1="0" y1="${padding}" x2="${w}" y2="${padding}" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4"/>
            <line x1="0" y1="${h/2}" x2="${w}" y2="${h/2}" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4"/>
            <line x1="0" y1="${h-padding}" x2="${w}" y2="${h-padding}" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4"/>
            ${yTickElements}
            ${showGen ? `<path d="${poly3}" fill="url(#g3)"/><path d="${path3}" fill="none" stroke="rgba(150,150,150,0.8)" stroke-width="2"/>` : ''}
            ${showMissed ? `<path d="${poly2}" fill="url(#g2)"/><path d="${path2}" fill="none" stroke="rgba(80,150,255,0.8)" stroke-width="2"/>` : ''}
            ${showCache ? `<path d="${poly1}" fill="url(#g1)"/><path d="${path1}" fill="none" stroke="rgba(0,200,100,0.8)" stroke-width="2"/>` : ''}
            
            ${(showGen?pts3:(showMissed?pts2:pts1)).map((p, i) => `<circle cx="${p.x}" cy="${p.y}" r="4" fill="var(--bg)" stroke="rgba(200,200,200,0.9)" stroke-width="2" class="chart-point-comp" data-idx="${i}" style="cursor:pointer; transition:all 0.2s;"/>`).join('')}
        </svg>
        <div id="${containerId}_tt" style="position:absolute; display:none; background:rgba(20,20,25,0.95); backdrop-filter:blur(10px); border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:10px 14px; font-size:12px; box-shadow:0 8px 32px rgba(0,0,0,0.5); pointer-events:none; z-index:100; white-space:nowrap; transition: left 0.1s, top 0.1s;"></div>
    `;
    
    const tt = document.getElementById(`${containerId}_tt`);
    container.querySelectorAll('.chart-point-comp').forEach(c => {
        c.addEventListener('mouseenter', (e) => {
            const idx = e.target.getAttribute('data-idx');
            const d = data[idx];
            c.setAttribute('r', '6');
            c.style.filter = 'drop-shadow(0 0 6px rgba(255,255,255,0.5))';
            tt.style.display = 'block';
            
            const pC = d.cached || 0;
            const pU = Math.max(0, (d.prompt || 0) - pC);
            const comp = Math.max(0, (d.tokens || 0) - pC - pU);
            
            tt.innerHTML = `
                <div style="color:var(--text-dim);margin-bottom:8px;font-weight:700;border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:6px;">${d.date}</div>
                ${showCache ? `<div style="display:flex; justify-content:space-between; width:160px; margin-bottom:4px;"><span style="color:var(--green)">命中缓存:</span> <span style="font-family:monospace">${fmtNum(pC)}</span></div>` : ''}
                ${showMissed ? `<div style="display:flex; justify-content:space-between; width:160px; margin-bottom:4px;"><span style="color:var(--blue)">未命中 Prompt:</span> <span style="font-family:monospace">${fmtNum(pU)}</span></div>` : ''}
                ${showGen ? `<div style="display:flex; justify-content:space-between; width:160px; margin-bottom:4px;"><span style="color:#aaa">生成 Output:</span> <span style="font-family:monospace">${fmtNum(comp)}</span></div>` : ''}
                <div style="display:flex; justify-content:space-between; width:160px; margin-top:6px; padding-top:6px; border-top:1px dashed rgba(255,255,255,0.1); font-weight:800; font-size:13px;"><span style="color:var(--text)">Total:</span> <span style="font-family:monospace">${fmtNum(d.tokens)}</span></div>
            `;
            
            let tx = parseFloat(c.getAttribute('cx')) + 15;
            let ty = parseFloat(c.getAttribute('cy')) - tt.clientHeight - 10;
            if (tx + 200 > container.clientWidth) tx = container.clientWidth - 210;
            if (ty < 0 || isNaN(ty)) ty = parseFloat(c.getAttribute('cy')) + 15;
            if (ty + 130 > container.clientHeight) ty = container.clientHeight - 135;
            tt.style.left = tx + 'px';
            tt.style.top = ty + 'px';
        });
        c.addEventListener('mouseleave', () => {
            c.setAttribute('r', '4');
            c.style.filter = 'none';
            tt.style.display = 'none';
        });
    });
}

function switchTab(tabId) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(tabId === 'pool' ? 'tabPool' : 'tabAnalytics').classList.add('active');
    
    document.getElementById('viewPool').style.display = tabId === 'pool' ? 'block' : 'none';
    document.getElementById('poolActions').style.display = tabId === 'pool' ? 'flex' : 'none';
    
    document.getElementById('viewAnalytics').style.display = tabId === 'analytics' ? 'block' : 'none';
    document.getElementById('analyticsActions').style.display = tabId === 'analytics' ? 'flex' : 'none';
    
    if(tabId === 'analytics') {
        loadAnalytics();
    }
}

function exportCSV() {
    window.open('/api/export-stats', '_blank');
}

let _analyticsData = null;

async function loadAnalytics(){
    const epFilter = document.getElementById('analyticsFilter').value || 'all';
    document.getElementById('tokenStatsOverview').innerHTML = '<div class="empty">加载中...</div>';
    
    const url = epFilter === 'all' ? '/api/token-stats' : '/api/token-stats?endpoint=' + encodeURIComponent(epFilter);
    const r = await api('GET', url);
    if(!r.today && r.today !== 0) {
        document.getElementById('tokenStatsOverview').innerHTML = '<div class="empty">加载失败</div>';
        return;
    }
    _analyticsData = r;
    
    const sel = document.getElementById('analyticsFilter');
    if (sel.options.length <= 1) {
        let opts = '<option value="all">全端点统计</option>';
        r.all_endpoints_list.forEach(e => { opts += `<option value="${esc(e)}">${esc(e)}</option>`; });
        sel.innerHTML = opts;
        sel.value = epFilter;
    }
    
    document.getElementById('tokenStatsOverview').innerHTML = `
        <div class="dash-stat"><div class="stat-icon">⚡</div><div class="num" style="color:var(--green)">${fmtNum(r.today)}</div><div class="label">今日消耗</div></div>
        <div class="dash-stat"><div class="stat-icon">📊</div><div class="num" style="color:var(--blue)">${fmtNum(r.last_3_days)}</div><div class="label">近 3 天</div></div>
        <div class="dash-stat"><div class="stat-icon">📅</div><div class="num" style="color:var(--yellow)">${fmtNum(r.last_7_days)}</div><div class="label">近 7 天</div></div>
        <div class="dash-stat"><div class="stat-icon">📈</div><div class="num" style="color:var(--accent-light)">${fmtNum(r.last_30_days)}</div><div class="label">近 30 天</div></div>
        <div class="dash-stat"><div class="stat-icon">🔥</div><div class="num" style="color:var(--orange)">${fmtNum(r.today_calls)}</div><div class="label">今日请求次数</div></div>
        <div class="dash-stat"><div class="stat-icon">🌍</div><div class="num" style="color:var(--orange)">${fmtNum(r.month_calls)}</div><div class="label">本月请求次数</div></div>
        <div class="dash-stat"><div class="stat-icon">💾</div><div class="num" style="color:var(--purple)">${r.today_cache_hit_rate}%</div><div class="label" style="display:flex; flex-direction:column;">今日缓存命中<span style="font-size:10px; color:var(--text-dim); margin-top:2px;">命中: ${fmtNum(r.today_cached)} / 未命: ${fmtNum(r.today_missed)}</span></div></div>
        <div class="dash-stat"><div class="stat-icon">🧠</div><div class="num" style="color:var(--purple)">${r.month_cache_hit_rate}%</div><div class="label" style="display:flex; flex-direction:column;">本月缓存命中<span style="font-size:10px; color:var(--text-dim); margin-top:2px;">命中: ${fmtNum(r.month_cached)} / 未命: ${fmtNum(r.month_missed)}</span></div></div>
    `;
    
    setTimeout(() => {
        drawSVGChart('tokenTodayChart', r.trend_today_hourly, {key: 'tokens', unit: 'Tokens'});
        drawSVGChart('missedTodayChart', r.trend_today_hourly, {key: 'missed', unit: 'Tokens'});
        drawSVGChart('callsTodayChart', r.trend_today_hourly, {key: 'calls', unit: '次'});
        drawCompositionChart('tokenTrendChart', r.trend_14d);
    }, 50);
    
    switchTblToday('tokens', true);
    switchTblMonth('tokens', true);
    
    // Default switchTrend to tokens is already handled in HTML. But just ensure UI state:
    switchTrend('tokens', true);
}

function renderTblData(containerId, data, key) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (!data || !data.length) {
        el.innerHTML = '<tr><td colspan="2" class="empty">暂无数据</td></tr>';
        return;
    }
    const maxVal = Math.max(...data.map(d => d[key])) || 1;
    el.innerHTML = data.map(d => `
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
            <td colspan="2" style="padding: 4px 0;">
                <div class="tbl-progress-container">
                    <div class="tbl-progress-bar" style="width: ${(d[key]/maxVal*100).toFixed(1)}%;"></div>
                    <div class="tbl-content">
                        <div><div style="font-size:10px; color:var(--text-dim); margin-bottom:2px;">${esc(d.endpoint)}</div><code>${esc(d.model)}</code></div>
                        <div style="text-align:right;">
                            <div style="font-family: monospace; font-weight:600;">${fmtNum(d[key])}</div>
                            <div style="font-size:9px; color:var(--purple); margin-top:2px;">命中率 ${d.cache_hit_rate||0}%</div>
                        </div>
                    </div>
                </div>
            </td>
        </tr>
    `).join('');
}

function updateCompChart() {
    if (_analyticsData) drawCompositionChart('tokenTrendChart', _analyticsData.trend_14d);
}

function switchTrend(type, skipRender) {
    document.getElementById('btnTrendToken').classList.toggle('active', type === 'tokens');
    document.getElementById('btnTrendMissed').classList.toggle('active', type === 'missed');
    document.getElementById('btnTrendCall').classList.toggle('active', type === 'calls');
    document.getElementById('tokenTodayChart').style.display = type === 'tokens' ? 'block' : 'none';
    document.getElementById('missedTodayChart').style.display = type === 'missed' ? 'block' : 'none';
    document.getElementById('callsTodayChart').style.display = type === 'calls' ? 'block' : 'none';
}

function switchTblToday(type, skipRender) {
    document.getElementById('btnTblTodayToken').classList.toggle('active', type === 'tokens');
    document.getElementById('btnTblTodayCall').classList.toggle('active', type === 'calls');
    document.getElementById('todayModelsTable').style.display = type === 'tokens' ? 'table-row-group' : 'none';
    document.getElementById('todayCallsTable').style.display = type === 'calls' ? 'table-row-group' : 'none';
    if (!skipRender && _analyticsData) {
        renderTblData(type === 'tokens' ? 'todayModelsTable' : 'todayCallsTable', _analyticsData.today_endpoints, type);
    } else if (skipRender && _analyticsData) {
        renderTblData('todayModelsTable', _analyticsData.today_endpoints, 'tokens');
        renderTblData('todayCallsTable', _analyticsData.today_endpoints, 'calls');
    }
}

function switchTblMonth(type, skipRender) {
    document.getElementById('btnTblMonthToken').classList.toggle('active', type === 'tokens');
    document.getElementById('btnTblMonthCall').classList.toggle('active', type === 'calls');
    document.getElementById('monthModelsTable').style.display = type === 'tokens' ? 'table-row-group' : 'none';
    document.getElementById('monthCallsTable').style.display = type === 'calls' ? 'table-row-group' : 'none';
    if (!skipRender && _analyticsData) {
        renderTblData(type === 'tokens' ? 'monthModelsTable' : 'monthCallsTable', _analyticsData.month_endpoints, type);
    } else if (skipRender && _analyticsData) {
        renderTblData('monthModelsTable', _analyticsData.month_endpoints, 'tokens');
        renderTblData('monthCallsTable', _analyticsData.month_endpoints, 'calls');
    }
}

let logAutoScroll = true;
const logContainer = document.getElementById('logContainer');
if (logContainer) {
    logContainer.addEventListener('scroll', () => {
        logAutoScroll = logContainer.scrollHeight - logContainer.clientHeight <= logContainer.scrollTop + 20;
    });
}
function addLogLine(entry) {
    if (!logContainer) return;
    const d = document.createElement('div');
    d.className = 'log-line';
    d.innerHTML = `<span class="log-time">[${entry.time}]</span> <span class="log-${entry.level}">[${entry.level}]</span> <span class="log-msg">${esc(entry.msg)}</span>`;
    logContainer.appendChild(d);
    if (logContainer.children.length > 300) logContainer.removeChild(logContainer.firstChild);
    if (logAutoScroll) logContainer.scrollTop = logContainer.scrollHeight;
}
let _lastLogId = 0;
async function pollLogs() {
    try {
        const logs = await api('GET', '/api/logs?since=' + _lastLogId);
        if (logs && logs.length > 0) {
            for (let entry of logs) {
                addLogLine(entry);
                _lastLogId = Math.max(_lastLogId, entry.id);
            }
        }
    } catch(err){}
    setTimeout(pollLogs, 2000);
}
pollLogs();

refresh();
setInterval(() => {
    refresh();
    loadChatLogs(chatLogsPage);
}, 3000);
let chatLogsPage = 0;
let currentChatLogs = [];
async function initChatLogs() {
  document.getElementById('clPrompt').textContent = '';
  document.getElementById('clCompletion').textContent = '';
  document.getElementById('clMeta').textContent = '';
  chatLogsPage = 0;
  await loadChatLogs(0);
}
initChatLogs();
async function loadChatLogs(page) {
  if (page < 0) return;
  const limit = 50;
  const offset = page * limit;
  const res = await api('GET', `/api/chat-logs?limit=${limit}&offset=${offset}`);
  if (!res.ok && res.error) { toast('加载失败', 'error'); return; }
  chatLogsPage = page;
  currentChatLogs = res.logs || [];
  
  const total = res.total || 0;
  const maxPage = Math.max(0, Math.ceil(total / limit) - 1);
  
  document.getElementById('clPageSpan').textContent = `${page + 1} / ${maxPage + 1}`;
  document.getElementById('clPrevBtn').disabled = page <= 0;
  document.getElementById('clNextBtn').disabled = page >= maxPage;
  
  const listEl = document.getElementById('chatLogsList');
  if (currentChatLogs.length === 0) {
    listEl.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-dim);">暂无日志记录</div>';
    return;
  }
  
  listEl.innerHTML = currentChatLogs.map((log, i) => `
    <div style="padding:8px 12px; border-bottom:1px solid var(--border); display:grid; grid-template-columns: 80px 1fr 1fr 80px; gap:8px; font-size:12px; cursor:pointer; transition:background 0.2s;" class="hover-bg" onclick="viewChatLog(${i})">
      <div style="color:var(--text-dim);">${log.timestamp.split(' ')[1]}</div>
      <div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${log.endpoint_name}">${log.endpoint_name}</div>
      <div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--accent-light);" title="${log.model}">${log.model}</div>
      <div style="color:var(--green);">${log.total_tokens}</div>
    </div>
  `).join('');
}
function viewChatLog(idx) {
  const log = currentChatLogs[idx];
  if (!log) return;
  document.getElementById('clPrompt').textContent = log.prompt || '';
  document.getElementById('clCompletion').textContent = log.completion || '';
  document.getElementById('clMeta').innerHTML = `<span style="color:var(--green)">${log.total_tokens} Tokens</span> <span style="margin-left:10px;color:var(--yellow)">${log.latency_ms}ms</span>`;
}
async function clearChatLogs() {
  if (!confirm('确定要清空所有对话日志记录吗？此操作不可逆。')) return;
  await api('DELETE', '/api/chat-logs');
  toast('已清空', 'success');
  loadChatLogs(0);
}

// Add CSS for hover
const style = document.createElement('style');
style.innerHTML = `
  .hover-bg:hover { background: rgba(255,255,255,0.05); }
`;
document.head.appendChild(style);

async function clearSysLogs() {
  if (!confirm('确定要清空系统日志吗？')) return;
  await api('DELETE', '/api/logs');
  document.getElementById('logContainer').innerHTML = '';
  toast('已清空日志', 'success');
}

async function clearTokenStats() {
  if (!confirm('确定要清空所有数据面板的 Token 统计记录吗？此操作不可逆。')) return;
  await api('DELETE', '/api/token-stats');
  toast('统计数据已清空', 'success');
  loadAnalytics();
}

</script>

<div id="testDrawer">
  <div class="drawer-header">
    <span id="testDrawerTitle">🧪 测试</span>
    <button class="btn btn-ghost btn-sm" onclick="closeTestDrawer()" style="padding: 2px 6px;">✖</button>
  </div>
  <div class="drawer-body">
    <input type="hidden" id="testTargetId" value="">
    <div id="testResult" class="test-result" style="margin-top:0; max-height:200px; display:none;"></div>
    <div class="test-input-row" style="display:flex; gap:8px;">
      <input type="text" id="testMsg" placeholder="测试消息..." value="用一句话介绍自己" style="flex:1">
      <input type="file" id="testImage" accept="image/*" style="display:none;" onchange="previewTestImage(this)">
      <button class="btn btn-ghost" onclick="document.getElementById('testImage').click()" title="上传图片测试" id="btnTestImage" style="padding:0 8px;font-size:16px;">🖼️</button>
      <button class="btn btn-primary" onclick="sendTest()">发送</button>
    </div>
  </div>
</div>
</body>

</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, code, data):
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            pass

    def _send_html(self, html):
        try:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            pass

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length:
                return json.loads(self.rfile.read(length))
            return {}
        except Exception:
            return {}

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send_html(GUI_HTML)
        elif self.path.startswith("/api/export-stats"):
            csv_data = token_tracker.export_csv()
            try:
                body = csv_data.encode("utf-8-sig")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=token_stats.csv")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except ConnectionError:
                pass
        elif self.path.startswith("/api/"):
            res = api_handler("GET", self.path, {})
            if len(res) == 3 and res[2] is True:
                code, stream_gen = res[0], res[1]
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for chunk in stream_gen:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except ConnectionError:
                    pass
            else:
                self._send_json(res[0], res[1])
        elif self.path.startswith(("/v1/", "/responses/")):
            res = api_handler("GET", self.path, {})
            self._send_json(res[0], res[1])
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        body = self._read_body()
        res = api_handler("POST", self.path, body)
        
        if len(res) == 3 and res[2] is True:
            code, stream_gen = res[0], res[1]
            try:
                self.send_response(code)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                
                for chunk in stream_gen:
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except ConnectionError:
                pass
        else:
            self._send_json(res[0], res[1])

    def do_PUT(self):
        body = self._read_body()
        res = api_handler("PUT", self.path, body)
        self._send_json(res[0], res[1])

    def do_DELETE(self):
        res = api_handler("DELETE", self.path, {})
        self._send_json(res[0], res[1])


def main():
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        try: sys.stdout.reconfigure(encoding='utf-8')
        except: pass
    port = 5100
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"\n  ⚡ API Pool 管理面板已启动")
    print(f"  🌐 管理面板访问: http://localhost:{port}")
    print(f"  🔗 客户端 Base URL: http://localhost:{port}/v1")
    print(f"  🔗 Responses API: http://localhost:{port}/v1/responses")
    print(f"  📋 已加载 {len(pool._endpoints)} 个端点")
    print(f"  🩺 健康检测: 启动时自动检测 + 每 {HEALTH_CHECK_INTERVAL}秒 复检\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()


if __name__ == "__main__":
    main()
