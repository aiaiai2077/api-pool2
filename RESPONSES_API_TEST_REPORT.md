# Responses API 全链路测试报告

## 测试信息

- 测试日期：2026-08-06
- 测试对象：`api_pool_server.py`
- 测试脚本：[tests/test_responses_api.py](tests/test_responses_api.py)
- 运行方式：`python tests\test_responses_api.py`
- 测试环境：本地 mock 上游，不访问真实计费 API

## 测试结果

| 结果 | 数量 |
| --- | --- |
| 通过 | 9 |
| 失败 | 0 |
| 总计 | 9 |

## 覆盖范围

1. 入站 `/v1/responses`，上游为 OpenAI 兼容 `chat/completions`
   - 文本输入、图片输入、`instructions` 转 system 消息
   - `reasoning.effort` 转 `reasoning_effort`
   - 非流式响应对象与真实 usage
   - 流式 SSE 生命周期与流式 usage
2. OpenAI 兼容上游工具调用
   - 非流式输出 `function_call` item
   - 流式输出 `response.function_call_arguments.delta/done`
3. Anthropic 上游工具调用
   - 非流式 `tool_use` 转 Responses `function_call`
   - 流式 Anthropic SSE 转 Responses 事件
4. 原生 `protocol="responses"` 上游
   - 请求转 `{base_url}/responses`
   - 非流式与流式双向转换
   - 通过 `/v1/chat/completions` 的回归验证
5. `store` / `previous_response_id`
   - 非流式多轮续接
   - 流式响应完成后继续续接
   - `GET /v1/responses/{id}` 检索
   - `DELETE /v1/responses/{id}` 删除
6. 上游流式结束标记
   - Responses 协议上游转 chat 流时输出 `finish_reason` 结束 chunk
   - Anthropic 上游 `message_stop` 同样补齐结束 chunk
   - 上游 `response.completed` 不带 usage 时仍保持单次结束
7. 错误处理
   - 缺 `input` 返回 400
   - 无效 `previous_response_id` 返回 400
8. 回归
   - 旧 `/v1/chat/completions` 非流式与流式
   - 健康检查、延迟测试调用点

## Claude API 专项检查

检查发现并修复了 Anthropic 协议分支的以下问题：

- `tools` 未转换：Chat/Responses 的 function tools 现在会转成 Anthropic `input_schema` 格式。
- `tool_choice` 未转换：`auto` / `any` / 指定函数现在会映射为 Anthropic `tool_choice`。
- assistant 消息里的 `tool_calls` 未转换：现在会生成 Anthropic `tool_use` content block。
- `role=tool` 消息未转换：现在会生成 `tool_result`，并合并进 user 消息。
- 连续 user 消息未合并：现在会合并，避免 Anthropic 格式拒绝。
- 补充 `top_k`、`stop_sequences`、`metadata.user_id` 映射。

Claude 测试覆盖：工具定义、`tool_choice`、多轮 tool_use/tool_result 转换、非流式与流式工具调用。

## 其他 API 问题检查

除 Claude 分支外，还检查并修复了以下问题：

- `/v1/chat/completions` 非流式响应之前丢失 tool_calls 且 usage 写死为 0，现在会透传真实 usage 和工具调用。
- Responses 多轮输入里的 `function_call` / `function_call_output` 之前会被降级为普通文本，现在会转成 chat 的 assistant tool_calls / tool role，Anthropic 侧继续转成 tool_use / tool_result。
- `metadata` 之前会透传给 OpenAI 兼容 chat 上游，可能导致严格供应商报错，现在只保留在 Responses 层和原生 Responses 上游。
- 无 tools 时不再发送 `tool_choice`，避免严格供应商报 400。
- 原生 Responses 流式兼容 `response.function_call_arguments.done` 作为工具调用收尾，不依赖 `output_item.done`。
- Responses 流式事件统一补充 `response_id` 字段，提升 SDK 兼容性。
- Responses / Anthropic 上游转 chat 流时补齐 `finish_reason` 结束 chunk，避免客户端（如 Hermes Studio）把正常结束误判为中途截断。
- 官方 `api.anthropic.com` 未带 `/v1` 的 base_url 会自动补全 `/v1/messages` 和 `/v1/models`。
- Responses `text.format` 结构化输出会映射为 chat 的 `response_format`，原生 Responses 上游会反向还原为 `text`。
- 本地 `store` 改为 SQLite 持久化（`responses_store.db`），重启不丢，容量仍限制为 200 条。

仍保留的边界：

- `reasoning` 仅透传 effort，未输出 summary。
- 未使用真实供应商 key 做网络实测。

## Hermes Studio 截断提示排查

Hermes Studio 界面报 `Error: Response remained truncated after 4 continuation attempts`，根因不在 Hermes：

- Hermes 通过自定义 Provider（`custom:api-pool`，Base URL `http://127.0.0.1:5100/v1`）调用 API Pool。
- 上游 OpenCode 走原生 Responses 协议时，API Pool 之前只转发内容 delta，随后直接发送 `usage` + `[DONE]`，没有带 `finish_reason` 的结束 chunk。
- Hermes 把这种流判为“中途截断”，自动续写，连续 4 次仍无结束标记后报错。
- 已修复：Responses / Anthropic 上游转 chat 流时在结束前补齐 `finish_reason`（文本 `stop`、工具 `tool_calls`、超限 `length`），并新增对应测试。
- 另外 OpenCode 的 `deepseek-v4-flash` 是推理模型，输出预算大部分消耗在 reasoning 上，正文通常很短；如希望正文更长，可在 Hermes 模型设置或请求里调大 `max_tokens` / `max_output_tokens`。

## 结论

上游与下游均已完成支持，且通过端到端闭环测试：

- 下游：客户端可通过 `/v1/responses` 使用 Responses API，包括流式、工具调用、usage、多轮记忆。
- 上游：支持 OpenAI 兼容、Anthropic、原生 Responses 三种协议，均可参与现有优先级调度与故障切换。
- 回归：旧 Chat Completions 网关行为未受影响。

## 当前边界

- 本地 `store` 已持久化到 SQLite，容量上限 200 条。
- 未实现 `reasoning` 摘要输出，仅透传 `reasoning.effort`。
- 未覆盖真实供应商网络环境，测试使用本地 mock 上游模拟协议行为。
