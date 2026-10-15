# vLLM 与 SGLang serve 启动参数详解大全

> 本文档整理了 vLLM 与 SGLang 两大主流 LLM 推理服务框架的 **全部 serve 启动命令参数** 及中文详解，内容基于官方最新文档（vLLM：docs.vllm.ai `vllm serve` CLI 参考；SGLang：docs.sglang.io Server Arguments），参数名、默认值与可选值均保持官方原样，说明为中文翻译。

## 一、概述

- **vLLM**：高性能 LLM 推理与服务库，以 PagedAttention、连续批处理（continuous batching）、前缀缓存等核心技术著称，提供 OpenAI 兼容 API 服务器。
- **SGLang**：面向生产级部署的高性能推理服务框架，以 RadixAttention 前缀缓存、多 GPU 并行（TP/DP/PP/EP）和极致吞吐著称，同样提供 OpenAI 兼容 API。

### 1.1 基本启动命令

```bash
# vLLM 启动 OpenAI 兼容服务器（默认端口 8000）
vllm serve <model_path_or_hf_id> --host 0.0.0.0 --port 8000

# SGLang 启动服务器（默认端口 30000）
python3 -m sglang.launch_server --model-path <model_path_or_hf_id> --host 0.0.0.0 --port 30000

# 查看全部参数
vllm serve --help
python3 -m sglang.launch_server --help
```

### 1.2 常用等价参数对照表

| 功能 | vLLM 参数 | SGLang 参数 |
|------|-----------|-------------|
| 模型路径 | `--model` | `--model-path` / `--model` |
| 分词器路径 | `--tokenizer` | `--tokenizer-path` |
| 服务端口 | `--port`（默认 8000） | `--port`（默认 30000） |
| 张量并行 | `--tensor-parallel-size, -tp` | `--tensor-parallel-size, -tp` |
| 流水线并行 | `--pipeline-parallel-size, -pp` | `--pipeline-parallel-size, -pp` |
| 数据并行 | `--data-parallel-size, -dp` | `--data-parallel-size, -dp` |
| 专家并行（MoE） | `--enable-expert-parallel` | `--ep-size` |
| 上下文最大长度 | `--max-model-len` | `--context-length` |
| 权重/激活数据类型 | `--dtype` | `--dtype` |
| 量化方法 | `--quantization, -q` | `--quantization` |
| KV 缓存数据类型 | `--kv-cache-dtype` | `--kv-cache-dtype` |
| GPU 显存占用比例 | `--gpu-memory-utilization` | `--mem-fraction-static` |
| 分块预填充 | `--max-num-batched-tokens` / `--enable-chunked-prefill` | `--chunked-prefill-size` |
| 最大运行请求数 | `--max-num-seqs` | `--max-running-requests` |
| 调度策略 | `--scheduler-delay-factor` | `--schedule-policy` |
| CUDA 图 | `--enforce-eager`（禁用）/ `--compilation-config` | `--cuda-graph-max-bs` / `--disable-cuda-graph` |
| API Key | `--api-key` | `--api-key` |
| 聊天模板 | `--chat-template` | `--chat-template` |
| 信任远程代码 | `--trust-remote-code` | `--trust-remote-code` |
| LoRA 支持 | `--enable-lora` + `--lora-modules` | `--lora-paths` |
| 指定 YAML 配置 | `--config` | `--config` |
| 工具调用解析器 | `--tool-call-parser` | `--tool-call-parser` |
| 推理内容解析 | `--reasoning-parser` | `--reasoning-parser` |

---

## 二、vLLM serve 启动参数详解

vLLM 通过 `vllm serve <model>` 命令启动 OpenAI 兼容的 API 服务器，可通过 `vllm serve --help` 查看全部启动参数。本节按配置组对所有参数进行分类详解，参数名保持英文原样，说明为中文翻译，默认值与可选值保持文档原样。

### 2.1 通用参数（General）

服务器启动相关的通用参数。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--headless` | 以 headless 模式运行，详见多节点数据并行文档 | `False` | — |
| `--api-server-count`, `-asc` | 运行的 API 服务器进程数，未指定时默认为 data_parallel_size | — | — |
| `--config` | 从 YAML 配置文件读取 CLI 选项 | — | — |
| `--grpc` | 在 --port 上启动 Python gRPC（SMG VllmEngine）服务器而非 HTTP OpenAI 兼容服务器；需 pip install vllm[grpc]；不能与 --grpc-port 组合 | `False` | — |
| `--disable-log-stats` | 禁用统计信息日志 | `False` | — |
| `--aggregate-engine-logging` | 使用数据并行时记录聚合而非单引擎统计信息 | `False` | — |
| `--fail-on-environ-validation`, `--no-fail-on-environ-validation` | 若设置，环境变量校验失败时引擎将抛出错误 | `False` | — |
| `--shutdown-timeout` | 关闭超时时间（秒）；0=立即中止，>0=等待 | `0` | — |
| `--gdn-prefill-backend` | 选择 GDN 预填充后端 | — | flashinfer / triton / cutedsl / aiter_flydsl |
| `--kda-prefill-backend` | 选择 KDA 预填充后端；flashkda 仅限 CUDA，fused 仅限 ROCm，auto 自动选择受支持的后端 | — | auto / triton / flashkda / flashinfer / fused |
| `--kda-decode-backend` | 选择 KDA 解码后端 | — | auto / native / flashinfer / triton |
| `--enable-log-requests`, `--no-enable-log-requests` | 启用请求信息日志（依赖日志级别：INFO 记录请求 ID、参数和 LoRA 请求；DEBUG 记录提示词输入如文本、token ID） | `False` | — |

### 2.2 前端配置（Frontend）

OpenAI 兼容前端服务器相关参数。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--lora-modules` | LoRA 模块配置，支持 name=path、JSON 或 JSON 列表格式 | — | — |
| `--chat-template` | 聊天模板的文件路径，或单行形式的模板内容 | — | — |
| `--chat-template-content-format` | 聊天模板中消息内容的渲染格式：string 渲染为字符串；openai 渲染为字典列表（类似 OpenAI schema） | `auto` | auto / openai / string |
| `--trust-request-chat-template`, `--no-trust-request-chat-template` | 是否信任请求中提供的聊天模板；为 False 时服务器始终使用 --chat-template 或 tokenizer 指定的模板 | `False` | — |
| `--trust-request-mm-kwargs`, `--no-trust-request-mm-kwargs` | 是否信任每请求多模态 kwargs（mm_processor_kwargs 和 media_io_kwargs）；会改变多模态预处理资源占用，仅在 API 客户端可信时启用 | `False` | — |
| `--default-chat-template-kwargs` | 传递给聊天模板渲染器的默认关键字参数，与请求级参数合并（请求值优先）；如 '{"enable_thinking": false}' 为 Qwen3/DeepSeek 默认关闭思考模式 | — | — |
| `--response-role` | 当 request.add_generation_prompt=true 时返回的角色名 | `assistant` | — |
| `--return-tokens-as-token-ids`, `--no-return-tokens-as-token-ids` | 指定 --max-logprobs 时，以 'token_id:{token_id}' 字符串形式表示单个 token，便于识别无法 JSON 编码的 token | `False` | — |
| `--enable-auto-tool-choice`, `--no-enable-auto-tool-choice` | 为支持的模型启用自动工具选择；用 --tool-call-parser 指定解析器 | `False` | — |
| `--exclude-tools-when-tool-choice-none`, `--no-exclude-tools-when-tool-choice-none` | 若设置，当 tool_choice='none' 时从提示词中排除工具定义 | `False` | — |
| `--tool-call-parser` | 根据所用模型选择工具调用解析器，将模型生成的工具调用解析为 OpenAI API 格式；--enable-auto-tool-choice 所必需 | — | — |
| `--tool-parser-plugin` | 工具解析器插件，注册的自定义解析器名可在 --tool-call-parser 中使用 | `""` | — |
| `--tool-strict-level` | 基于结构化标签的工具调用的服务端下限：auto 跟随请求的 tool choice 与逐工具 strict；function 约束工具调用外壳（标记和函数名）；parameter 额外固定参数 schema | `auto` | auto / function / parameter |
| `--tool-server` | 逗号分隔的 host:port 列表（IPv4/IPv6/主机名），或 demo 使用内置演示工具（浏览器和 Python 代码解释器，注意 demo 工具的安全风险） | — | — |
| `--max-log-len` | 日志中打印的最大提示词字符数或 token ID 数；None 表示不限制 | — | — |
| `--enable-prompt-tokens-details`, `--no-enable-prompt-tokens-details` | 若为 True，在 usage 中启用 prompt_tokens_details | `False` | — |
| `--enable-per-request-metrics`, `--no-enable-per-request-metrics` | 若为 True，在 API 响应中包含每请求计时指标 | `False` | — |
| `--enable-server-load-tracking`, `--no-enable-server-load-tracking` | 若为 True，在应用状态中启用 server_load_metrics 跟踪 | `False` | — |
| `--enable-force-include-usage`, `--no-enable-force-include-usage` | 若为 True，每个请求都包含 usage | `False` | — |
| `--sse-keep-alive-interval` | /v1/chat/completions 或 /v1/completions 流式响应空闲时（排队、预填充或 token 间）每多少秒发送一次 SSE keep-alive 注释行，防止反向代理/隧道因读超时关闭连接；0 表示完全禁用 | `0` | — |
| `--enable-tokenizer-info-endpoint`, `--no-enable-tokenizer-info-endpoint` | 启用 /tokenizer_info 端点，可能暴露聊天模板等 tokenizer 配置 | `False` | — |
| `--enable-log-outputs`, `--no-enable-log-outputs` | 若为 True，记录模型输出（生成内容）；需 --enable-log-requests；输出文本和结束原因记于 INFO，输出 token ID 记于 DEBUG | `False` | — |
| `--enable-log-deltas`, `--no-enable-log-deltas` | 若为 False，不记录输出增量；仅在设置 --enable-log-outputs 时相关 | `True` | — |
| `--cohere-is-reasoning-model`, `--no-cohere-is-reasoning-model` | 仅 Cohere /cohere/v2/chat：模型是否为推理型 Command 系列；True 时助手思维链以 thinking 内容块（或流上的 content-* 事件）呈现 | `True` | — |
| `--cohere-format` | 仅 --tokenizer-mode cohere：渲染的 Cohere 提示词格式；选错格式会静默产生模型未训练过的提示词（常见表现为有文本但无引用/工具调用/思考块） | `cmd4` | cmd4 / cmd3 |
| `--anthropic-disabled-thinking-effort` | 仅 Anthropic /v1/messages：thinking 为 disabled 的请求所用的 reasoning_effort；none 关闭思考，low 适配总是思考或拒绝 none 的模型（如 GLM-5.3、gpt-oss） | `auto` | auto / high / low / max / medium / none / xhigh |
| `--log-error-stack`, `--no-log-error-stack` | 若为 True，记录错误响应的堆栈跟踪 | `False` | — |
| `--tokens-only`, `--no-tokens-only` | 若为 True，仅启用 Tokens In&lt;&gt;Out 端点；用于 Disaggregated Everything 部署 | `False` | — |
| `--enable-scale-out`, `--no-enable-scale-out` | 若为 True，在 vllm serve 上注册扩缩容端点（/render、/derender、/inference/v1/generate 和 /inference/v1/abort_requests） | `False` | — |
| `--fingerprint-mode` | 控制响应中 system_fingerprint 字段：full 编码版本、非平凡并行度（tp/pp/dp/ep）和 8 字符配置哈希；hash 去掉并行度；custom 使用 --fingerprint-value 的字面值；none 省略该字段 | `full` | custom / full / hash / none |
| `--fingerprint-value` | --fingerprint-mode=custom 时使用的字面指纹字符串 | — | — |
| `--host` | 主机名 | — | — |
| `--port` | 端口号 | `8000` | — |
| `--grpc-port` | 在该端口启用 Rust 前端的附加 gRPC 推理与控制服务；需 VLLM_USE_RUST_FRONTEND=1 vllm serve，HTTP 仍在 --port 上；不能与 Python gRPC 的 --grpc 组合 | — | — |
| `--data-parallel-supervisor-port` | 多端口外部 LB 模式下聚合健康端点的 HTTP 端口 | `9256` | — |
| `--dp-supervisor-probe-interval-s` | 多端口外部 LB 模式下聚合健康探测间隔（秒） | `5.0` | — |
| `--dp-supervisor-probe-timeout-s` | 多端口外部 LB 模式下子健康探测连接失败时重试等待时间（秒） | `5.0` | — |
| `--dp-supervisor-probe-failure-threshold` | 多端口外部 LB 模式下子健康探测判定失败前的连续连接错误重试次数 | `3` | — |
| `--uds` | Unix 域套接字路径；设置后 host 和 port 参数被忽略 | — | — |
| `--uvicorn-log-level` | uvicorn 日志级别 | `info` | critical / debug / error / info / trace / warning |
| `--disable-uvicorn-access-log`, `--no-disable-uvicorn-access-log` | 禁用 uvicorn 访问日志 | `False` | — |
| `--disable-access-log-for-endpoints` | 从 uvicorn 访问日志中排除的端点路径列表（逗号分隔），如 "/health,/metrics,/ping"，用于减少健康检查等高频端点的日志噪音 | — | — |
| `--allow-credentials`, `--no-allow-credentials` | 允许凭证 | `False` | — |
| `--allowed-origins` | 允许的源；默认禁用跨域浏览器访问 | `[]` | — |
| `--allowed-methods` | 允许的方法 | `['*']` | — |
| `--allowed-headers` | 允许的头部 | `['*']` | — |
| `--api-key` | 若提供，服务器将要求请求头中携带该密钥；仅保护 /v1、/v2 和 /inference 前缀下的端点，勿单独依赖其保证安全 | — | — |
| `--ssl-keyfile` | SSL 密钥文件路径 | — | — |
| `--ssl-certfile` | SSL 证书文件路径 | — | — |
| `--ssl-ca-certs` | CA 证书文件 | — | — |
| `--enable-ssl-refresh`, `--no-enable-ssl-refresh` | SSL 证书文件变更时刷新 SSL 上下文 | `False` | — |
| `--ssl-cert-reqs` | 是否要求客户端证书（参见标准库 ssl 模块） | `0` | — |
| `--ssl-ciphers` | HTTPS 的 SSL 密码套件（仅 TLS 1.2 及以下），如 'ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305' | — | — |
| `--root-path` | 应用位于基于路径的路由代理之后时的 FastAPI root_path | — | — |
| `--middleware` | 附加到应用的 ASGI 中间件（可多次指定），值为导入路径；函数以 @app.middleware('http') 添加，类以 app.add_middleware() 添加 | `[]` | — |
| `--enable-request-id-headers`, `--no-enable-request-id-headers` | 若指定，API 服务器将在响应中添加 X-Request-Id 头 | `False` | — |
| `--disable-fastapi-docs`, `--no-disable-fastapi-docs` | 禁用 FastAPI 的 OpenAPI schema、Swagger UI 和 ReDoc 端点 | `False` | — |
| `--h11-max-incomplete-event-size` | h11 解析器允许的不完整 HTTP 事件（头部或主体）最大字节数，缓解头部滥用 | `4194304` | — |
| `--h11-max-header-count` | h11 解析器允许的单个请求最大 HTTP 头数量，缓解头部滥用 | `256` | — |
| `--enable-offline-docs`, `--no-enable-offline-docs` | 为隔离网络（air-gapped）环境启用离线 FastAPI 文档，使用 vLLM 内置的静态资源 | `False` | — |
| `--enable-flash-late-interaction`, `--no-enable-flash-late-interaction` | 若设置，在 API 服务器进程中于 GPU 上运行池化打分 MaxSim，可显著提升 late-interaction 打分性能；禁用时引擎侧使用参考 MaxSim 路径而非融合 Triton 内核 | `True` | — |

### 2.3 模型配置（ModelConfig）

模型相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--model` | 要使用的 Hugging Face 模型名称或路径；未指定 served_model_name 时也用作指标输出中 model_name 标签的内容 | `Qwen/Qwen3-0.6B` | — |
| `--runner` | 要使用的模型运行器类型；每个 vLLM 实例仅支持一种模型运行器 | `auto` | auto / draft / generate / pooling |
| `--convert` | 使用适配器转换模型，最常见用途是将文本生成模型适配用于池化任务 | `auto` | auto / classify / embed / none |
| `--tokenizer` | 要使用的 Hugging Face tokenizer 名称或路径；未指定时使用模型名称或路径 | — | — |
| `--tokenizer-mode` | tokenizer 模式：auto 对 Mistral 模型优先用 mistral_common 否则用 hf；mistral 始终用 mistral_common；cohere 用 cohere_melody 渲染聊天模板；kimi_k3 用 HF tokenizer 但以 Kimi K3 的 Python XTML 编码渲染；支持插件自定义 | `auto` | auto / cohere / deepseek_v32 / deepseek_v4 / hf / inkling / kimi_k3 / mistral |
| `--trust-remote-code`, `--no-trust-remote-code` | 下载模型和 tokenizer 时信任远程代码（如来自 HuggingFace） | `False` | — |
| `--dtype` | 模型权重和激活的数据类型：auto 对 FP32/FP16 模型用 FP16、对 BF16 模型用 BF16；half 即 FP16（AWQ 量化推荐）；bfloat16 平衡精度与范围；float32 为 FP32 | `auto` | auto / bfloat16 / float / float16 / float32 / half |
| `--seed` | 用于可复现性的随机种子；必须设置全局种子，否则不同张量并行 worker 会采样出不同 token 导致结果不一致 | `0` | — |
| `--hf-config-path` | 要使用的 Hugging Face config 名称或路径；未指定时使用模型名称或路径 | — | — |
| `--allowed-local-media-path` | 允许 API 请求从服务器文件系统指定目录读取本地图像或视频；有安全风险，仅应在可信环境启用 | `""` | — |
| `--allowed-media-domains` | 若设置，仅允许属于该域的媒体 URL 用于多模态输入 | — | — |
| `--revision` | 要使用的模型版本，可为分支名、标签名或 commit id；未指定时使用默认版本 | — | — |
| `--code-revision` | Hugging Face Hub 上模型代码的版本，可为分支名、标签名或 commit id | — | — |
| `--tokenizer-revision` | Hugging Face Hub 上 tokenizer 的版本，可为分支名、标签名或 commit id | — | — |
| `--max-model-len` | 模型上下文长度（提示词+输出）；未指定时从模型配置自动推导；支持 k/m/g 人类可读格式，-1 或 auto 表示自动选择能装入显存的最大长度 | — | — |
| `--quantization`, `-q` | 权重量化方法；为 None 时先检查模型配置文件的 quantization_config，仍为 None 则视为未量化并按 dtype 决定权重数据类型 | — | — |
| `--quantization-config` | 用户侧量化配置，携带按层类别（linear、moe）的规格与忽略模式；quantization 为在线量化简写时自动填充 | — | — |
| `--allow-deprecated-quantization`, `--no-allow-deprecated-quantization` | 是否允许已弃用的量化方法 | `False` | — |
| `--enforce-eager`, `--no-enforce-eager` | 是否始终使用 eager 模式 PyTorch；为 True 时禁用 CUDA 图并始终以 eager 模式执行模型；等价于 -cc.mode=none -cc.cudagraph_mode=none | `False` | — |
| `--enable-return-routed-experts`, `--no-enable-return-routed-experts` | 捕获并返回 routed-experts 辅助输出 | `False` | — |
| `--return-sampling-mask`, `--no-return-sampling-mask` | 是否返回每个样本的后处理 token 支持信息 | `False` | — |
| `--max-logprobs` | 指定 logprobs 时返回的最大对数概率数量；-1 表示不限制（可能 OOM） | `20` | — |
| `--logprobs-mode` | logprobs 与 prompt_logprobs 返回内容：raw 表示任何 logit 处理器之前的值，processed 表示应用全部处理器（含 temperature 和 top_k/top_p）之后的值；prompt_logprobs 两者结果相同 | `raw_logprobs` | processed_logits / processed_logprobs / raw_logits / raw_logprobs |
| `--use-fp64-gumbel`, `--no-use-fp64-gumbel` | Gumbel-max 及等价指数竞速采样是否使用 FP64（而非 FP32）随机噪声；FP64 保留低尾采样事件但多数 GPU 上吞吐显著下降 | `False` | — |
| `--enable-trace-replay`, `--no-enable-trace-replay` | 是否允许请求设置 trace_decode_token_ids 以按预定 token 序列解码（仍计算真实 logprobs）；保留用于调试和 RL 工作流 | `False` | — |
| `--disable-sliding-window`, `--no-disable-sliding-window` | 是否禁用滑动窗口；为 True 时禁用模型滑动窗口功能并截断到滑动窗口大小；模型不支持时忽略 | `False` | — |
| `--disable-cascade-attn`, `--no-disable-cascade-attn` | 已弃用且无效（级联注意力已移除），将在 v0.33.0 移除 | — | — |
| `--skip-tokenizer-init`, `--no-skip-tokenizer-init` | 跳过 tokenizer 和 detokenizer 初始化；要求输入提供有效 prompt_token_ids 且 prompt 为 None，生成输出仅含 token id | `False` | — |
| `--enable-prompt-embeds`, `--no-enable-prompt-embeds` | 若为 True，允许通过 prompt_embeds 传入文本嵌入；传入错误形状嵌入可能导致引擎崩溃，仅为可信用户启用 | `False` | — |
| `--served-model-name` | API 中使用的模型名；可提供多个名称，响应 model 字段取第一个；也用于 Prometheus 指标 model_name 标签（多名称时取第一个） | — | — |
| `--config-format` | 要加载的模型配置格式：auto 先尝试 mistral 格式再尝试 hf 格式 | `auto` | auto / hf / mistral |
| `--hf-token` | 用于远程文件 HTTP bearer 授权的 token；为 True 时使用 hf auth login 生成的 token | — | — |
| `--hf-overrides` | 若为字典，包含要转发给 Hugging Face config 的参数；若为可调用对象，则调用它更新 config | `{}` | — |
| `--model-class-overrides` | 覆盖一个或多个架构使用的模型类，架构名映射到 "module:class" 目标；仅用于开发和调试 | `{}` | — |
| `--pooler-config` | 池化配置，控制池化模型输出池化的行为 | — | — |
| `--generation-config` | 生成配置的文件夹路径；auto 从模型路径加载；vllm 不加载生成配置而使用 vLLM 默认值；生成配置中指定 max_new_tokens 时将作为全局输出 token 上限 | `auto` | — |
| `--override-generation-config` | 覆盖或设置生成配置，如 {"temperature": 0.5}；与 auto 配合时与模型默认配置合并，与 vllm 配合时仅使用覆盖参数 | `{}` | — |
| `--enable-sleep-mode`, `--no-enable-sleep-mode` | 为引擎启用休眠模式（仅支持 cuda 和 hip 平台） | `False` | — |
| `--sleep-mode-offload-cudagraph`, `--no-sleep-mode-offload-cudagraph` | 休眠时将 CUDA 图内存备份到 CPU，唤醒时原地恢复；配合 enable_sleep_mode、cumem 后端和 CUDA 图生效 | `False` | — |
| `--sleep-preserve-parameter-names` | 跨 level-2 休眠保留的参数名 glob 模式，用于训练器省略的冻结 Engram 表权重和缩放；draft.&lt;name&gt; 用于草稿模型私有参数，draft.* 用于全部 | `[]` | — |
| `--enable-cumem-allocator`, `--no-enable-cumem-allocator` | 启用自定义 cumem 分配器，以利用多节点 NVLink 支持等高级 GPU 内存分配特性；休眠模式自动启用该分配器 | `False` | — |
| `--enable-nccl-comm-suspend`, `--no-enable-nccl-comm-suspend` | 休眠期间释放 NCCL 通信器内存（ncclCommSuspend/ncclCommResume）；实验性 | `False` | — |
| `--model-impl` | 使用哪种模型实现：auto 优先 vLLM 实现，无 vLLM 实现时回退 Transformers；transformers 使用 Transformers 实现；terratorch 使用 TerraTorch 实现 | `auto` | auto / terratorch / transformers / vllm |
| `--logits-processors` | 一个或多个 logits 处理器的完整类名或类定义 | — | — |
| `--io-processor-plugin` | 模型启动时加载的 IOProcessor 插件名 | — | — |
| `--renderer-num-workers` | 渲染器线程池中的工作线程数；异步渲染路径（如 vllm serve 启动的 OpenAI 兼容服务器）用它并行化分词、聊天模板渲染和多模态预处理；离线 LLM 入口为同步路径，该设置无效 | `1` | — |

### 2.4 权重加载配置（LoadConfig）

模型权重加载相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--load-format` | 模型权重加载格式：auto 优先 safetensors 回退 pytorch bin；dummy 用随机值初始化（用于 profiling）；sharded_state 从预分片检查点加载，支持张量并行模型高效加载；tensorizer 用 CoreWeave 的 tensorizer 库快速加载等 | `auto` | auto / pt / safetensors / instanttensor / ipc_cache / npcache / dummy / tensorizer / runai_streamer / runai_streamer_sharded / sharded_state / mistral / modelexpress |
| `--download-dir` | 下载和加载权重的目录，默认为 Hugging Face 默认缓存目录 | — | — |
| `--safetensors-load-strategy` | safetensors 权重加载策略：None 用内存映射（lazy）加载并在 NFS 上自动预取；lazy 内存映射按需加载；eager 将整个文件预先读入 CPU 内存（推荐网络文件系统如 Lustre、NFS）；prefetch 预读到 OS 页缓存；torchao 重建 torchao 张量子类 | — | eager / lazy / prefetch / torchao / None |
| `--safetensors-prefetch-num-threads` | 启用 safetensors 预取时用于将检查点文件预取到 OS 页缓存的工作线程数 | `8` | — |
| `--safetensors-prefetch-block-size` | 每次 safetensors 检查点文件预取的读取字节数（支持人类可读格式如 1k、25.6k） | `16777216` | — |
| `--model-loader-extra-config` | 模型加载器的额外配置，将传递给所选 load_format 对应的模型加载器 | `{}` | — |
| `--ignore-patterns` | 加载模型时忽略的模式列表；默认 original/**/* 以避免重复加载 llama 检查点 | `['original/**/*']` | — |
| `--use-tqdm-on-load`, `--no-use-tqdm-on-load` | 加载模型权重时是否启用 tqdm 进度条 | `True` | — |
| `--pt-load-map-location` | 加载 pytorch 检查点的 map_location，如 {"": "cuda"} 或 {"cuda:1": "cuda:0"}（命令行传字典时字符串需双重引号） | `cpu` | — |

### 2.5 注意力配置（AttentionConfig）

vLLM 注意力机制相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--attention-backend` | 使用的注意力后端；使用 "auto" 或 None 自动选择 | — | — |

### 2.6 Mamba 配置（MambaConfig）

Mamba SSM 后端相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--mamba-backend` | 使用的 Mamba SSU 后端 | `MambaBackendEnum.TRITON` | — |
| `--mamba-ssu-algorithm` | 配合 FlashInfer 后端使用的选择性状态更新算法；None 使用 FlashInfer 的 auto 算法；强制指定的算法须被当前 GPU、状态 dtype 和解码模式支持 | — | auto / horizontal / simple / vertical / None |
| `--enable-mamba-cache-stochastic-rounding`, `--no-enable-mamba-cache-stochastic-rounding` | 将 SSM 状态写入 fp16 缓存时启用随机舍入；用随机位消除舍入偏差，可改善长序列数值稳定性 | `False` | — |
| `--mamba-cache-philox-rounds` | 随机舍入随机数生成的 Philox PRNG 轮数；0 使用 Triton 默认，更大值随机性更好但更耗算力 | `0` | — |

### 2.7 结构化输出配置（StructuredOutputsConfig）

引擎结构化输出相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--reasoning-parser` | 根据所用模型选择推理内容解析器，将推理内容解析为 OpenAI API 格式；none 表示禁用（同不设置） | `""` | — |
| `--reasoning-parser-plugin` | 可动态加载和注册的推理解析器插件路径 | `""` | — |

### 2.8 并行配置（ParallelConfig）

分布式执行相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--distributed-executor-backend` | 分布式模型 worker 后端：ray 或 mp（多进程）；流水线并行大小与张量并行大小乘积不超过可用 GPU 数时使用 mp，否则报错；TPU 平台仅支持 Ray | — | external_launcher / mp / ray / uni |
| `--pipeline-parallel-size`, `-pp` | 流水线并行组数 | `1` | — |
| `--master-addr` | distributed_executor_backend 为 mp 时多节点分布式推理的主地址 | `127.0.0.1` | — |
| `--master-port` | distributed_executor_backend 为 mp 时多节点分布式推理的主端口 | `29501` | — |
| `--nnodes`, `-n` | distributed_executor_backend 为 mp 时多节点分布式推理的节点数 | `1` | — |
| `--node-rank`, `-r` | distributed_executor_backend 为 mp 时多节点分布式推理的节点序号 | `0` | — |
| `--distributed-timeout-seconds` | 分布式操作超时（秒），传递给 torch.distributed.init_process_group 的 timeout 参数；None 用 PyTorch 默认（NCCL 为 600s） | — | — |
| `--cpu-distributed-timeout-seconds` | CPU 通信组超时（秒）；None 用 PyTorch 默认（gloo 为 1800s） | — | — |
| `--numa-bind`, `--no-numa-bind` | 为 GPU worker 子进程启用 NUMA 绑定；默认将 worker 绑定到其 GPU 的 NUMA 本地 CPU 和内存，PCT 能力的 Xeons 还会自动绑定 SKU 的 PCT 优先核 | `False` | — |
| `--numa-bind-nodes` | 每个 GPU worker 绑定的 NUMA 节点，每个可见 GPU 一个，如 [0, 0, 1, 1]；未设置且 numa_bind=True 时自动检测 GPU-NUMA 拓扑；值传递给 numactl --membind 和 --cpunodebind | — | — |
| `--numa-bind-cpus` | 每个 GPU worker 绑定的可选 CPU 列表，每个可见 GPU 一项，如 ["0-3", "4-7"]；设置后用 numactl --physcpubind 替代 --cpunodebind，适用于绑定 PCT 等高频核的自定义策略 | — | — |
| `--device-ids` | 要使用的逗号分隔物理 GPU 设备 ID 或 UUID（如 --device-ids "2,3,5,7"）；避免设置 CUDA_VISIBLE_DEVICES 以保留 GPU-NIC 亲和性和 DeepGEMM 所需的完整拓扑可见性；对 Ray 执行器无效 | — | — |
| `--tensor-parallel-size`, `-tp` | 张量并行组数 | `1` | — |
| `--decode-context-parallel-size`, `-dcp` | 对解码 KV缓存分片的 rank 数；DCP 不扩展进程世界大小；无 PCP 时复用 TP rank，有 PCP 时跨 PCP 轴或整个 TP x PCP 块 | `1` | — |
| `--dcp-comm-backend` | 解码上下文并行（DCP）的通信后端：ag_rs 为 AllGather+ReduceScatter；a2a 为部分输出+LSE 的全交换并用 Triton 内核合并，可将 MLA 模型每层 NCCL 调用从 3 次减到 2 次 | — | a2a / ag_rs / None |
| `--dcp-q-replicate`, `--no-dcp-q-replicate` | 在每个 DCP 组内复制 MLA 查询投影，使解码可跳过每步的查询 all-gather；代价是组内每个 rank 冗余计算该投影 | — | — |
| `--dcp-kv-cache-interleave-size` | 使用 DCP 时 KV缓存存储的交错大小；已被 cp_kv_cache_interleave_size 取代，PCP 完全支持后将弃用 | `1` | — |
| `--cp-kv-cache-interleave-size` | 使用 DCP 时 KV缓存的交错大小：1 为 token 级对齐（token i 存于 dcp_rank i mod dcp_world_size）；block_size 为块级对齐；须能整除 block_size；未指定时按 NIXL 传输需求自动解析 | — | — |
| `--prefill-context-parallel-size`, `-pcp` | 分割预填充序列计算的 rank 数；PCP 扩展进程世界大小但不增加 KV缓存分片数 | `1` | — |
| `--data-parallel-size`, `-dp` | 数据并行组数；MoE 层按张量并行、预填充上下文并行和数据并行大小的乘积分片 | `1` | — |
| `--data-parallel-rank`, `-dpn` | 本实例的数据并行 rank；设置后为 MoE 数据并行部署启用外部负载均衡模式；不支持非 MoE 模型（应启动独立 vLLM 实例） | — | — |
| `--data-parallel-start-rank`, `-dpr` | 辅助节点的起始数据并行 rank | — | — |
| `--data-parallel-size-local`, `-dpl` | 本节点运行的数据并行副本数 | — | — |
| `--data-parallel-address`, `-dpa` | 数据并行集群头节点地址 | — | — |
| `--data-parallel-rpc-port`, `-dpp` | 数据并行 RPC 通信的固定端口；所有节点必须使用相同端口 | — | — |
| `--data-parallel-backend`, `-dpb` | 数据并行后端，mp 或 ray | `mp` | — |
| `--data-parallel-hybrid-lb`, `--no-data-parallel-hybrid-lb`, `-dph` | 是否使用混合 DP LB 模式（仅在线服务且 data_parallel_size>0 时适用）：vLLM 在本地数据并行 rank 间负载均衡，vLLM 节点/副本间由外部 LB 负载均衡；需与 --data-parallel-start-rank 显式配合 | `False` | — |
| `--data-parallel-external-lb`, `--no-data-parallel-external-lb`, `-dpe` | 是否使用外部 DP LB 模式（仅在线服务且 data_parallel_size>0 时适用）；适用于 Kubernetes 的 one-pod-per-rank 宽 EP 部署；仅支持 MoE 部署 | `False` | — |
| `--data-parallel-multi-port-external-lb`, `-dpm` | 运行节点本地 supervisor，为每个本地数据并行 rank 启动一个外部 LB API 服务器，并在 supervisor 端口上暴露聚合健康状态 | `False` | — |
| `--enable-hc-sp`, `--no-enable-hc-sp` | 为 HyperConnection（HC）计算启用序列并行；当前由 Qwen4Exp 在 CUDA 上 TP&gt;1 且 PP=1 时支持（TP=1 时忽略）；不改变 MoE 并行 | `False` | — |
| `--enable-expert-parallel`, `--no-enable-expert-parallel`, `-ep` | MoE 层使用专家并行而非张量并行 | `False` | — |
| `--enable-batch-sharded-sampling`, `--no-enable-batch-sharded-sampling` | 跨张量并行 rank 使用分片采样，每个 rank 采样批次的一个切片而非全部；要求 tensor_parallel_size&gt;1、max_num_seqs 不小于 TP、max_logprobs 非负 | — | — |
| `--enable-ep-weight-filter`, `--no-enable-ep-weight-filter` | 专家并行激活时加载模型跳过非本地专家权重，每个 rank 仅从磁盘读取自己的专家分片，可大幅减少含逐专家权重张量的 MoE 模型（如 DeepSeek、Mixtral、Kimi-K2.5）的存储 I/O | `False` | — |
| `--all2all-backend` | MoE 专家并行通信的 All2All 后端：deepep_high_throughput/deepep_low_latency 用 DeepEP 高吞吐/低延迟内核；moonep 为带动态冗余专家的均衡 EP（NVLink）；flashinfer_nvlink_two_sided 用 mnnvl 双侧内核等 | `allgather_reducescatter` | allgather_reducescatter / deepep_high_throughput / deepep_low_latency / deepep_v2 / flashinfer_all2allv / flashinfer_nvlink_one_sided / flashinfer_nvlink_two_sided / moonep / mori_high_throughput / mori_low_latency / naive / nixl_ep / pplx |
| `--enable-dbo`, `--no-enable-dbo` | 为模型执行器启用双重批次重叠（dual batch overlap） | `False` | — |
| `--enable-shm-tensor-arena`, `--no-enable-shm-tensor-arena` | 引擎到 worker 广播中的大型 CPU 张量（如多模态 pixel_values）经零拷贝共享内存 arena 路由而非给每个读者发副本；实验性，多模态大图+张量并行场景收益最大 | `False` | — |
| `--ubatch-size` | 微批次（ubatch）大小 | `0` | — |
| `--enable-elastic-ep`, `--no-enable-elastic-ep` | 为 DP/EP 启用基于无状态 NCCL 组的弹性专家并行 | `False` | — |
| `--elastic-ep-max-dp-size` | 弹性专家并行支持的最大数据并行大小 | — | — |
| `--dbo-decode-token-threshold` | 纯解码批次的双重批次重叠阈值；token 数超过该阈值时使用微批处理，否则单批处理 | `32` | — |
| `--dp-sync-interval` | DP 完成同步 all-reduce 之间的步数；所有 DP rank 必须一致 | `16` | — |
| `--dbo-prefill-token-threshold` | 含一个或多个预填充批次的双重批次重叠阈值；token 数超过该阈值时使用微批处理，否则单批处理 | `512` | — |
| `--disable-nccl-for-dp-synchronization`, `--no-disable-nccl-for-dp-synchronization` | 强制 DP 同步逻辑使用 Gloo 而非 NCCL 进行 all-reduce；启用异步调度时默认 True，否则 False | — | — |
| `--enable-eplb`, `--no-enable-eplb` | 为 MoE 层启用专家并行负载均衡（EPLB） | `False` | — |
| `--eplb-config` | 专家并行负载均衡配置 | `EPLBConfig(window_size=1000, step_interval=3000, num_redundant_experts=0, log_balancedness=False, use_async=True, policy='default', ...)` | — |
| `--expert-placement-strategy` | MoE 层专家放置策略：linear 连续放置（4 专家 2 rank 时 rank 0 得 [0,1]）；round_robin 轮询放置（rank 0 得 [0,2]），可改善无冗余专家的分组专家模型的负载均衡 | `linear` | linear / round_robin |
| `--max-parallel-loading-workers` | 多批次顺序加载模型时的最大并行加载 worker 数，避免大模型张量并行时的内存 OOM | — | — |
| `--ray-workers-use-nsight`, `--no-ray-workers-use-nsight` | 是否使用 nsight 分析 Ray worker | `False` | — |
| `--disable-custom-all-reduce`, `--no-disable-custom-all-reduce` | 禁用自定义 all-reduce 内核并回退到 NCCL | `False` | — |
| `--worker-cls` | 要使用的 worker 类全名；为 "auto" 时根据平台确定 | `auto` | — |
| `--worker-extension-cls` | 要使用的 worker 扩展类全名，被 worker 类动态继承，用于向 worker 类注入新属性和方法以在 collective_rpc 调用中使用 | `""` | — |
| `--enable-fault-tolerance`, `--no-enable-fault-tolerance` | 启用容错以进行详细错误恢复，如缩容故障 DPEngineCore | `False` | — |
| `--fault-tolerance-config` | 容错配置 | `FaultToleranceConfig(engine_recovery_timeout_sec=120)` | — |

### 2.9 缓存配置（CacheConfig）

KV缓存相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--block-size` | 连续缓存块的 token 数；接受 None（表示使用默认值） | — | — |
| `--gpu-memory-utilization`, `--device-memory-utilization` | 模型执行器可用的 GPU 显存比例（0 到 1），如 0.5 表示 50%；为单实例限制，不影响同 GPU 上的其他 vLLM 实例；非 GPU 安装上控制对应设备内存利用率 | `0.92` | — |
| `--kv-cache-memory-bytes` | 每块 GPU 的 KV缓存大小（字节）；默认 None 时按 gpu_memory_utilization 自动推断；设置后忽略 gpu_memory_utilization，提供更细粒度控制；支持人类可读格式 | — | — |
| `--kv-cache-dtype` | KV缓存存储的数据类型；auto 使用模型数据类型；CUDA 11.8+ 支持 fp8（=fp8_e4m3）和 fp8_e5m2；ROCm 支持 fp8；DeepSeekV3.2 等部分模型默认 fp8，可设 bfloat16 改用 BF16 | `auto` | auto / bfloat16 / float16 / fp8 / fp8_ds_mla / fp8_e4m3 / fp8_e5m2 / fp8_inc / fp8_per_token_head / int4_per_token_head / int8_per_token_head / nvfp4 / nvfp4_4over6 / nvfp4_ds_mla / turboquant_3bit_nc / turboquant_4bit_nc / turboquant_k3v4_nc / turboquant_k8v4 / ultraquant_4bit |
| `--num-gpu-blocks-override` | 要使用的 GPU 块数，覆盖 profiling 得到的 num_gpu_blocks；None 时不生效；用于测试抢占 | — | — |
| `--enable-prefix-caching`, `--no-enable-prefix-caching` | 是否启用前缀缓存 | — | — |
| `--prefix-caching-hash-algo` | 前缀缓存的哈希算法：sha256 用 Pickle 序列化（默认，最安全）；sha256_cbor 用规范 CBOR 序列化（可跨语言复现）；xxhash 用 xxHash 128 位更快但非加密安全（多租户环境注意碰撞风险）；xxhash_cbor 结合 CBOR 与 xxHash | `sha256` | sha256 / sha256_cbor / xxhash / xxhash_cbor |
| `--prefix-cache-retention-interval` | 保留滑动窗口和 Mamba 前缀缓存检查点之间的 token 间隔；0 仅保留语义检查点（最新重放边界和共享前缀交汇点）；正值额外按间隔保留（须为调度器块大小的倍数）；None 密集保留 | `0` | — |
| `--kv-cache-dtype-skip-layers` | 跳过 KV缓存量化的层模式，接受层索引（如 0、2、4）或注意力类型名（如 sliding_window） | `[]` | — |
| `--kv-sharing-fast-prefill`, `--no-kv-sharing-fast-prefill` | 在 YOCO 等 KV 共享场景下允许部分层跳过预填充对应的 token，为合格层启用注意力元数据覆盖（如 Gemma3n） | `False` | — |
| `--swa-bounded-replay`, `--no-swa-bounded-replay` | 将支持模型的滑动窗口 KV（DeepSeek-V4.1）排除出前缀缓存，命中后仅重算命中的最后窗口重建；最后一个 KV 源层之后的层也仅对每请求尾部窗口执行 eager 预填充；需要模型运行器 V2 | `True` | — |
| `--mamba-cache-dtype` | Mamba 缓存（conv 和 ssm 状态）的数据类型；auto 时从模型配置推断 | `auto` | auto / bfloat16 / float16 / float32 |
| `--mamba-ssm-cache-dtype` | Mamba 缓存中 ssm 状态的数据类型（conv 状态仍由 mamba_cache_dtype 控制）；auto 时由 mamba_cache_dtype 决定 | `auto` | auto / bfloat16 / float16 / float32 |
| `--mamba-block-size` | Mamba 缓存连续块的 token 数；仅在前缀缓存启用时可设置 | — | — |
| `--prefix-match-unit` | 前缀缓存命中的最小 token 边界；每该数量 token 计算一次缓存键，可设得比物理 KV缓存块更细（如混合模型 1024 块下取 32），只要每个 KV缓存组的 block_size 能被其整除；等同 KV缓存代码中的 hash_block_size | — | — |
| `--mamba-cache-mode` | 已弃用并将在 v0.33.0 移除，无效果：Mamba 前缀缓存跟随 --enable-prefix-caching | — | align / none |
| `--enable-mamba-shared-prefix-checkpoint`, `--no-enable-mamba-shared-prefix-checkpoint` | 在共享前缀交汇处（观察到 EAGLE/MTP 兄弟恢复的位置）额外注册 Mamba align 检查点；仅在前缀缓存启用、EAGLE 作用于 Mamba 组且 prefix match unit 小于 Mamba 块大小时生效 | `False` | — |
| `--replayssm-buffer-len` | Mamba2 的 ReplaySSM 逻辑历史长度 B；Triton 用 B 个物理行，FlashInfer 用 B+T（T 为目标验证长度）；Kimi-K3 投机解码不使用 B | `16` | — |
| `--use-replayssm`, `--no-use-replayssm` | 使用 ReplaySSM Mamba2 解码内核：缓存近期 SSM 输入并跳过每步全量状态存储，仅在 flush 时写回检查点；需 Triton 或 FlashInfer mamba 后端 | `False` | — |
| `--kv-offloading-size` | KV缓存卸载缓冲区大小（GiB）；TP&gt;1 时为所有 TP rank 之和；默认 None 表示不启用；设置后用 kv_offloading_backend 启用 CPU 卸载 | — | — |
| `--kv-offloading-backend` | KV缓存卸载后端；仅在设置 kv_offloading_size 后激活 | `native` | lmcache / native |

### 2.10 卸载配置（OffloadConfig）

模型权重卸载以减少 GPU 显存使用的相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--offload-backend` | 权重卸载后端：auto 按子配置非默认值选择（offload_group_size&gt;0 选 prefetch，cpu_offload_gb&gt;0 选 uva）；uva 为统一虚拟寻址（UVA）零拷贝卸载；prefetch 为基于层组的异步预取卸载 | `auto` | auto / prefetch / uva |
| `--cpu-offload-gb` | 每块 GPU 卸载到 CPU 的空间（GiB），可视为虚拟扩大显存（如 24GB GPU 设 10 相当于 34GB）；部分模型每次前向都需从 CPU 内存即时加载，要求快速 CPU-GPU 互连；使用 UVA 零拷贝访问 | `0` | — |
| `--cpu-offload-params` | CPU 卸载目标参数名段集合，未匹配的参数不卸载；为空时非选择性卸载直到 cpu_offload_gb 定义的上限；按段精确匹配（"experts" 匹配 "mlp.experts.w2_weight" 但 "expert"、"w2" 不匹配） | `set()` | — |
| `--offload-group-size` | 每 N 层分为一组，卸载每组最后 offload_num_in_group 层（如 group_size=8、num_in_group=2 卸载 6,7,14,15,...）；0 表示禁用；与 cpu_offload_gb 不同，使用显式异步预取隐藏传输延迟 | `0` | — |
| `--offload-num-in-group` | 每组卸载的层数，必须 &lt;= offload_group_size | `1` | — |
| `--offload-prefetch-step` | 向前预取的层数；更大值隐藏更多延迟但占用更多显存 | `1` | — |
| `--offload-params` | 预取卸载目标参数名段集合，未匹配的参数不卸载；为空时卸载每个被卸载层的全部参数；按段匹配（"w13_weight" 匹配 "mlp.experts.w13_weight" 但不匹配 "mlp.experts.w13_weight_scale"） | `set()` | — |

### 2.11 多模态配置（MultiModalConfig）

多模态模型行为控制相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--language-model-only`, `--no-language-model-only` | 若为 True，通过将所有模态上限设为 0 禁用全部多模态输入，等价于对每个模态设 --limit-mm-per-prompt 0 | `False` | — |
| `--limit-mm-per-prompt` | 每个提示词每种模态允许的最大输入项数与选项；默认每模态 999；支持纯数量旧格式、含 count/num_frames/width/height 的可配置格式及混合格式 | `{}` | — |
| `--enable-mm-embeds`, `--no-enable-mm-embeds` | 若为 True，允许传入多模态嵌入（LLM 类为 multi_modal_data 张量输入；服务器为 "*_embeds" 类型内容）；与 limit-mm-per-prompt 0 配合可跳过数量校验且不加载编码器；传入错误形状嵌入可能导致崩溃 | `False` | — |
| `--media-io-kwargs` | 按模态传递给媒体输入处理的额外参数，如 '{"video": {"num_frames": 40}}' | `{}` | — |
| `--mm-processor-kwargs` | 转发给模型多模态处理器（如图像处理器）的参数，覆盖 transformers.AutoProcessor.from_pretrained 的默认值；可用项取决于所运行的模型（如 Phi-3-Vision 的 {"num_crops": 4}） | — | — |
| `--mm-processor-cache-gb` | 多模态处理器缓存大小（GiB），避免重复处理历史输入；该缓存按 API 进程和引擎核心进程复制，总用量为该值乘以 (api_server_count + data_parallel_size)；设 0 完全禁用（不推荐） | `4` | — |
| `--mm-processor-cache-type` | 多模态预处理器/映射器缓存类型：shm 为共享内存 FIFO 缓存；lru 为镜像 LRU 缓存 | `lru` | lru / shm |
| `--mm-hasher-algorithm` | 多模态输入缓存使用的哈希算法；FIPS 合规部署用 sha256 或 sha512 | `blake3` | blake3 / sha256 / sha512 |
| `--mm-shm-cache-max-object-size-mb` | 多模态处理器共享内存缓存中单个对象的大小上限（MiB）；仅在 mm_processor_cache_type 为 shm 时有效 | `128` | — |
| `--mm-encoder-only`, `--no-mm-encoder-only` | 启用时跳过模型的语言组件；通常仅在分离式 Encoder 进程中有效 | `False` | — |
| `--mm-encoder-tp-mode` | 多模态编码器推理的 TP 优化方式：weights 跨 TP rank 切分每层权重（默认 TP 行为）；data 切分批输入数据并行处理，每个 rank 持有完整权重（不同于请求级 DP）；编码器不支持时回退 weights | `weights` | data / weights |
| `--mm-encoder-attn-backend` | 视觉 Transformer 多模态编码器注意力后端的可选覆盖；接受 AttentionBackendEnum 中的任意值（如 FLASH_ATTN） | — | — |
| `--mm-encoder-attn-dtype` | ViT 编码器注意力的可选 dtype 覆盖；设为 fp8 时通过 FlashInfer cuDNN 后端启用 FP8 量化，无 scale 文件时自动使用动态缩放 | — | fp8 / None |
| `--mm-encoder-fp8-scale-path` | 含 ViT 编码器注意力逐层 FP8 Q/K/V 缩放的 JSON 文件路径；提供时（配合 mm_encoder_attn_dtype="fp8"）使用静态缩放，省略时使用动态缩放 | — | — |
| `--mm-encoder-fp8-scale-save-path` | 动态 FP8 缩放下 amax 历史缓冲区满后将校准缩放保存到该文件，后续运行可作为 mm_encoder_fp8_scale_path 使用 | — | — |
| `--mm-encoder-fp8-scale-save-margin` | 自动保存时乘到缩放上的安全裕量；大于 1 的值留出余量，防止超出校准集的更大激活溢出 FP8 范围 | `1.5` | — |
| `--interleave-mm-strings`, `--no-interleave-mm-strings` | 在使用 --chat-template-content-format=string 时启用完全交错的多模态提示词支持 | `False` | — |
| `--skip-mm-profiling`, `--no-skip-mm-profiling` | 启用时跳过多模态内存 profiling，引擎初始化时仅用语言主干模型 profiling；加快启动但需用户自行估计多模态编码器激活和嵌入缓存的峰值内存 | `False` | — |
| `--video-pruning-rate` | 每个视频要剪枝的视频 token 比例，范围 [0,1)；大于 0 时启用剪枝，算法由 video_pruning_method 选择 | — | — |
| `--video-pruning-method` | video_pruning_rate&gt;0 时应用的视频 token 剪枝算法：evs 为高效视频采样（Efficient Video Sampling）；vidcom2 为 Video Compression Commander | `evs` | evs / vidcom2 |
| `--mm-tensor-ipc` | 多模态张量的 IPC（进程间通信）方式：direct_rpc 用 msgspec 序列化经 RPC；torch_shm 用 torch.multiprocessing 共享内存零拷贝 IPC | `direct_rpc` | direct_rpc / torch_shm |
| `--mm-processor-device` | HF 多模态处理器运行图像/视频变换的设备（等价于 --mm-processor-kwargs '{"device": ...}'）；仅对接受 device 参数的 HF fast（torchvision 后端）处理器生效；auto 仅在编码器实例且 torch_shm 可承载设备张量时用加速器，否则为 cpu | `auto` | auto / cpu |
| `--mm-ipc-gpu-memory-gb` | 在 API 服务器（前端）进程为 GPU 侧多模态工作（如硬件视频解码）在引擎设备上预留的显存量（GiB）；该预算从引擎 KV缓存内存中划出，前端 GPU 解码路径经阻塞字节计数信号量申请；0 表示禁用 | `0` | — |
| `--mm-device-do-normalize`, `--no-mm-device-do-normalize` | 将多模态预处理中的 do_normalize 计算移到 ViT 之前并交由设备执行，节省 CPU 计算 | — | — |

### 2.12 LoRA 配置（LoRAConfig）

LoRA 相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--enable-lora`, `--no-enable-lora` | 若为 True，启用 LoRA 适配器处理 | — | — |
| `--max-loras` | 单个批次中最大 LoRA 数量 | `1` | — |
| `--max-lora-rank` | 最大 LoRA rank | `16` | 1 / 8 / 16 / 32 / 64 / 128 / 256 / 320 / 512 |
| `--lora-dtype` | LoRA 数据类型；auto 时使用基础模型 dtype | `auto` | — |
| `--enable-tower-connector-lora`, `--no-enable-tower-connector-lora` | 若为 True，为多模态模型的 tower（视觉编码器）和 connector 启用 LoRA 支持；实验性，目前仅支持 Qwen VL 系列等部分多模态模型 | `False` | — |
| `--max-cpu-loras` | CPU 内存中存储的最大 LoRA 数量；必须 &gt;= max_loras | — | — |
| `--fully-sharded-loras`, `--no-fully-sharded-loras` | 默认仅一半 LoRA 计算随张量并行分片；启用后使用完全分片层，高序列长度、大 rank 或大 TP 时可能更快 | `False` | — |
| `--lora-target-modules` | 将 LoRA 限制到特定模块后缀（如 ["o_proj", "qkv_proj"]）；None 时使用所有支持的 LoRA 模块，便于部署时控制应用 LoRA 的模块以调优性能 | — | — |
| `--default-mm-loras` | 模态到 LoRA 模型路径的映射字典；仅适用于多模态模型，某模态出现时总是需要激活 LoRA 的场景；当前每提示词仅支持一个 LoRA 适配器 | — | — |
| `--specialize-active-lora`, `--no-specialize-active-lora` | 是否按活跃 LoRA 适配器数量构建 lora 内核网格；为 True 时为不同活跃数量（max_loras 内的 2 的幂）分别捕获 CUDA 图，以启动时间和内存换可变 LoRA 使用模式的性能 | `False` | — |
| `--enable-mixed-moe-lora-format`, `--no-enable-mixed-moe-lora-format` | 若为 True，强制使用通用 2D MoE LoRA 包装器（FusedMoEWithLoRA），使 2D 与 3D 格式 MoE LoRA 适配器可在同一部署中服务；仅对 MoE 模型有意义 | `False` | — |
| `--enable-moe-shared-loras`, `--no-enable-moe-shared-loras` | 若为 True，以 shared-outer 布局加载 MoE 专家适配器：gate/up（w1/w3）的 lora_A 与 down（w2）的 lora_B 跨所有专家共享（专家维为 1），内核时广播到专家数；仅对使用该布局的 MoE 模型有意义 | `False` | — |
| `--max-lora-cls-labels` | LoRA 分类头的最大输出尺寸；默认基础分类头大小 | — | — |

### 2.13 日志配置（LoggingConfig）

vLLM 日志相关配置。可通过 `--logging-config` 传入 JSON 对象，或用点分参数传单个字段（如 `--logging-config.log_level DEBUG`）。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--logging-config` | 日志配置（vllm.config.LoggingConfig） | — | — |
| `--log-level` | --logging-config.log_level 的快捷方式；两者都指定时覆盖该字段 | — | CRITICAL / ERROR / WARNING / INFO / DEBUG / NOTSET |
| `--log-config-file` | 已弃用，将在 v0.33.0 移除；请改用 --logging-config.pylogging_config_file | — | — |

### 2.14 可观测性配置（ObservabilityConfig）

指标与追踪等可观测性相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--show-hidden-metrics-for-version` | 启用自指定版本起被隐藏的已弃用 Prometheus 指标，如 --show-hidden-metrics-for-version=0.7 作为迁移到新指标期间的临时方案 | — | — |
| `--otlp-traces-endpoint` | OpenTelemetry trace 发送的目标 URL | — | — |
| `--collect-detailed-traces` | 为指定模块收集详细 trace；涉及可能昂贵或阻塞的操作，可能影响性能；仅在设置 --otlp-traces-endpoint 时有意义 | — | all / model / worker / None |
| `--per-request-spec-decode-metrics` | 在响应 metrics.speculative_decoding 中包含每请求投机解码接受指标：none 禁用；summary 增加平均接受长度、草稿接受率和分步直方图；detailed 额外记录有序的逐步接受/提议数组；仅单序列请求报告 | `none` | detailed / none / summary |
| `--kv-cache-metrics`, `--no-kv-cache-metrics` | 启用 KV缓存驻留指标（生存期、空闲时间、复用间隙）；使用采样最小化开销；需启用日志统计（即未设 --disable-log-stats） | `False` | — |
| `--kv-cache-metrics-sample` | KV缓存指标采样率 (0.0, 1.0]；默认 0.01 即 1% 的块 | `0.01` | — |
| `--custom-histogram-buckets` | 自定义 Prometheus 直方图桶边界（桶族键到严格递增正有限上界列表的 JSON 映射）；已知族含 request_latency、time_to_first_token、inter_token_latency 等；每多一个桶会增加一条时间序列 | — | — |
| `--cudagraph-metrics`, `--no-cudagraph-metrics` | 启用 CUDA 图指标（填充/未填充 token 数、运行时 cudagraph 分发模式及每个日志间隔的观测频率） | `False` | — |
| `--enable-layerwise-nvtx-tracing`, `--no-enable-layerwise-nvtx-tracing` | 启用逐层 NVTX 追踪：追踪模型每层/模块执行并向 nvtx range 标记附加输入输出形状等信息；与启用的 CUDA 图不兼容 | `False` | — |
| `--enable-mfu-metrics`, `--no-enable-mfu-metrics` | 启用 MFU（模型算力利用率，Model FLOPs Utilization）指标 | `False` | — |
| `--enable-logging-iteration-details`, `--no-enable-logging-iteration-details` | 启用迭代详情日志：vLLM EngineCore 将记录上下文/生成请求数和 token 数及迭代 CPU 耗时 | `False` | — |
| `--jit-monitor-mode` | 如何处理预热后的 JIT 编译事件 | `warn` | error / warn |
| `--jit-monitor-verbose`, `--no-jit-monitor-verbose` | 记录每个被监控 JIT 编译的运行时详情；可能产生大量日志并增加开销，仅用于调试 | `False` | — |

### 2.15 调度器配置（SchedulerConfig）

调度器相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--max-num-batched-tokens` | 单次迭代可处理的最大 token 数 | — | — |
| `--max-num-scheduled-tokens` | 调度器单次迭代可发出的最大 token 数；通常等于 max_num_batched_tokens，但模型可能向批次附加 token（如投机解码）时可更小 | — | — |
| `--max-num-seqs` | 单次迭代处理的最大序列数 | — | — |
| `--max-num-active-seqs` | 调度器允许进入 RUNNING 的最大请求数；只降低 RUNNING 占用而不缩小 runner（每请求缓冲和 CUDA 图捕获）容量；必须 &lt;= max_num_seqs；None 保持当前行为 | — | — |
| `--max-num-queued-reqs` | 同时在途（等待或运行）的最大请求数，None 无限制；达到上限时新请求被 HTTP 503 拒绝以便客户端重试其他实例；在 API 服务器进程统计其路由的所有 DP rank 的在途请求 | — | — |
| `--max-num-queued-tokens` | 当前处于预填充阶段请求的最大总提示 token 数，None 无限制；达到上限时新请求被 HTTP 503 拒绝，属 TTFT QoS 机制（设为 目标TTFT x 预填充吞吐） | — | — |
| `--long-prefill-token-threshold` | 分块预填充下提示超过该 token 数视为长请求；0 表示不限制（默认）；请求是批次中唯一请求时不设限（无其他请求可被饿死） | `0` | — |
| `--long-prefill-token-threshold-adaptive`, `--no-long-prefill-token-threshold-adaptive` | 将有效长预填充阈值下限设为 token 预算的公平份额：max_num_batched_tokens 除以排队和运行请求数；仅在 long_prefill_token_threshold 非零时适用 | `False` | — |
| `--scheduling-policy` | 调度策略：fcfs 先来先服务；priority 按给定优先级处理（值小先处理），同优先级按到达时间决胜 | `fcfs` | fcfs / priority |
| `--enable-chunked-prefill`, `--no-enable-chunked-prefill` | 若为 True，预填充请求可按剩余 max_num_batched_tokens 分块 | — | — |
| `--disable-chunked-mm-input`, `--no-disable-chunked-mm-input` | 若为 True 且启用分块预填充，不部分调度多模态项：确保混合提示词中图像 token 只能整体调度（如 TTTT+IIII 分两步各调度完整部分） | `False` | — |
| `--scheduler-cls` | 要使用的调度器类；默认 vllm.v1.core.sched.scheduler.Scheduler，可为类或 "mod.custom_class" 形式的路径 | — | — |
| `--scheduler-reserve-full-isl`, `--no-scheduler-reserve-full-isl` | 若为 True，调度器在接纳新请求前检查完整输入序列长度是否放得下 KV缓存（而非仅第一块），防止分块预填充下的过度接纳和 KV缓存抖动 | `True` | — |
| `--watermark` | 接纳等待或被抢占请求进入运行队列时保持空闲的 KV缓存块比例，避免显存紧张时频繁 KV缓存驱逐和请求反复抢占；范围 [0.0, 1.0)，0.0（默认）禁用 | `0.0` | — |
| `--prefill-schedule-interval` | 数据并行部署时每 N 个引擎步才接纳一次新预填充请求（跨 DP rank 对齐），以更好平衡每步前向时间 | `1` | — |
| `--disable-hybrid-kv-cache-manager`, `--no-disable-hybrid-kv-cache-manager` | 若为 True，即使存在全注意力和滑动窗口注意力等多种注意力层，也为所有注意力层分配相同大小的 KV缓存；None 时按环境和启动配置决定 | — | — |
| `--async-scheduling`, `--no-async-scheduling` | 异步调度帮助避免 GPU 利用率空隙，改善延迟和吞吐；为 False 时禁用 | — | — |
| `--stream-interval` | 流式传输的间隔（缓冲大小，以 token 数计）；1 每个 token 立即发送更流畅，较大值（如 10）批量发送以减少主机开销并可能提升吞吐 | `1` | — |

### 2.16 编译配置（CompilationConfig）

torch.compile 与 CUDA 图捕获相关配置。含 PassConfig（控制自定义融合/变换 pass），其余分三部分：顶层编译控制、CUDA 图捕获、Inductor 编译。可用 `-cc.parameter=argument` 简写形式追加编译参数。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--cudagraph-capture-sizes` | 捕获 CUDA 图的尺寸：None 从 vllm 配置推断；list[int] 按给定指定 | — | — |
| `--max-cudagraph-capture-size` | 最大 CUDA 图捕获尺寸；指定 cudagraph_capture_sizes 时取其中最大值（或校验一致性）；未指定时按 [1,2,4]+range(8,256,8)+range(256,max+1,16) 模式生成，默认上限 512（Blackwell 数据中心 GPU 为 1024），避免小 max_num_seqs 下 OOM | — | — |

### 2.17 内核配置（KernelConfig）

内核选择与预热行为相关配置。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--ir-op-priority` | 前向过程中 dispatch/lowering 的 vLLM IR op 优先级；平台默认值在 VllmConfig post_init 时自动追加 | `IrOpPriorityConfig(rms_norm=[], fused_add_rms_norm=[], gelu_and_mul_sparse=[])` | — |
| `--enable-flashinfer-autotune`, `--no-enable-flashinfer-autotune` | 若为 True，在内核预热期间运行 FlashInfer 自动调优 | — | — |
| `--monokernel`, `--no-monokernel` | 若为 True，在模型 MonoKernel 上运行解码步的合格层：单个 launch 运行一个解码层的全部或大部分（MI355X 上的 DeepSeek-V4.1）；无法运行的部署启动时拒绝 | `False` | — |
| `--moe-backend` | MoE 专家计算内核后端：auto 按模型和硬件自动选择；triton 用 Triton 融合 MoE 内核；deep_gemm 用 DeepGEMM（仅 FP8 块量化）；marlin 用 Marlin（权重量化）；emulation 反量化到 BF16/FP16 运行等 | `auto` | aiter / aiter_mega_moe / aiter_triton_mxfp4_bf16 / auto / b12x / batched_triton / cutlass / deep_gemm / deep_gemm_mega_moe / emulation / flashinfer_b12x / flashinfer_cutedsl / flashinfer_cutlass / flashinfer_moe_ep_cutedsl / flashinfer_moe_ep_mega_deep_gemm / flashinfer_trtllm / flydsl / hpc / humming / marlin / rdna3 / triton / triton_unfused |
| `--linear-backend` | 线性层 GEMM 内核后端：请求后端未实现的层类型自动选择；cutlass 用 CUTLASS 内核；torch 用 PyTorch 原生 scaled_mm；marlin 用 Marlin 内核；emulation 为慢速反量化仿真（仅测试）等 | `auto` | aiter / auto / b12x / conch / cutlass / deep_gemm / emulation / exllama / fbgemm / flashinfer_b12x / flashinfer_cudnn / flashinfer_cutedsl / flashinfer_cutlass / flashinfer_trtllm / humming / machete / marlin / torch / triton / xpu / xpu_woq |
| `--sparse-indexer-topk-backend` | DSA 稀疏索引器解码 top-k 内核后端：auto 使用 cooperative-&gt;persistent-&gt;per_row 链；deep_select 仅 SM100a/SM103a；torch 为普通 torch.topk 调试参考实现等 | `auto` | aiter / auto / cooperative / deep_select / flashinfer / per_row / persistent / torch |

### 2.18 VllmConfig 综合配置（VllmConfig）

包含全部 vLLM 相关配置的数据类，用于简化各配置在代码中的传递。

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--speculative-config`, `-sc` | 投机解码配置（vllm.config.SpeculativeConfig） | — | — |
| `--spec-method` | 使用的投机方法名；提供 model 参数时自动检测投机方法类型，未提供时必须提供方法名；ngram 方法需考虑 prompt_lookup_max 和 prompt_lookup_min 配置 | — | bailing_hybrid_mtp / bailing_hybrid_v3_mtp / custom_class / deepseek_mtp / dflash / dots3_note_mtp / draft_model / dspark / eagle / eagle3 / ernie_mtp / exaone4_5_mtp / exaone_moe_mtp / extract_hidden_states / gemma4_mtp / glm4_moe_lite_mtp / glm4_moe_mtp / glm5_next_mtp / glm_ocr_mtp / hy_v3_mtp / hy_v4_mtp / inkling_mtp / kimi_k3_mtp / longcat_flash_mtp / mimo_mtp / mimo_v2_mtp / minimax_m3_mtp / mtp / nemotron_h_mtp / ngram / ngram_gpu / pangu_ultra_moe_mtp / qwen3_5_mtp / qwen3_next_mtp / qwen4_exp_mtp / step3p5_mtp / suffix / None |
| `--spec-model` | 草稿模型、eagle head 或附加权重的名称 | — | — |
| `--spec-tokens` | 投机 token 数；有草稿模型配置时默认取其中数量，否则必须提供 | — | — |
| `--watermark-config` | 文本水印配置（vllm.config.WatermarkConfig） | — | — |
| `--diffusion-config`, `-dc` | 扩散 LLM（dLLM）配置（vllm.config.DiffusionConfig） | — | — |
| `--kv-transfer-config` | 分布式 KV缓存传输配置（vllm.config.KVTransferConfig） | — | — |
| `--kv-events-config` | 事件发布配置（vllm.config.KVEventsConfig） | — | — |
| `--ec-transfer-config` | 分布式 EC 缓存传输配置（vllm.config.ECTransferConfig） | — | — |
| `--ec-manager-config` | 自定义编码器缓存管理器配置 | `EncoderCacheManagerConfig(encoder_cache_manager_cls=None, manager_config={})` | — |
| `--compilation-config`, `-cc` | 模型的 torch.compile 和 CUDA 图捕获配置；可用 -cc.parameter=argument 简写（如 -cc.mode=3 等价 -cc='{"mode":3}'），可指定如 {"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]} | `{'mode': None, 'backend': 'inductor', 'cudagraph_mode': None, ...}` | — |
| `--attention-config`, `-ac` | 注意力配置（vllm.config.AttentionConfig） | `AttentionConfig(backend=None, minimax_m3_msa_decode_backend='triton', ...)` | — |
| `--engram-config` | N-gram 嵌入存储与分片设置（vllm.config.EngramConfig） | — | — |
| `--reasoning-config` | 推理模型配置（vllm.config.ReasoningConfig） | — | — |
| `--kernel-config` | 内核配置（vllm.config.KernelConfig） | `KernelConfig(ir_op_priority=..., moe_backend='auto', linear_backend='auto', ...)` | — |
| `--additional-config` | 指定平台的附加配置；不同平台支持不同配置，须确保所用平台有效，内容必须可哈希 | `{}` | — |
| `--structured-outputs-config` | 结构化输出配置 | `StructuredOutputsConfig(backend='auto', reasoning_parser='', enable_in_reasoning=False, ...)` | — |
| `--aux-output-config` | 执行辅助输出配置 | `AuxOutputConfig(enable_return_routed_experts=False, max_bytes=None)` | — |
| `--profiler-config` | 性能分析配置 | `ProfilerConfig(profiler=None, torch_profiler_dir='', proton_backend=None, ...)` | — |
| `--optimization-level` | 优化级别：以启动时间换性能，-O0 启动时间最好，-O3 性能最好 | `2` | — |
| `--performance-mode` | 运行时性能模式：interactivity 偏好小批次低端到端每请求延迟（细粒度 CUDA 图、延迟导向内核）；throughput 偏好高并发聚合吞吐（更大 CUDA 图、更激进批处理、吞吐导向内核） | `balanced` | balanced / interactivity / throughput |
| `--weight-transfer-config` | RL 训练期间权重传输配置（vllm.config.WeightTransferConfig） | — | — |


## 三、SGLang serve 启动参数详解

SGLang 通过 `python3 -m sglang.launch_server --model-path <model>` 启动推理服务，可通过 `--help` 查看全部启动参数，并支持 `--config` 指定 YAML 配置文件（命令行参数会覆盖配置文件中的值）。常用技巧：多 GPU 张量并行加 `--tp 2`，数据并行加 `--dp 2`；出现显存溢出（OOM）时可调小 `--mem-fraction-static`，长 prompt 预填充 OOM 时可调小 `--chunked-prefill-size`；启用 FP8 权重量化加 `--quantization fp8`，FP8 KV缓存量化加 `--kv-cache-dtype fp8_e4m3`。以下按官方文档分类整理全部启动参数。

### 3.1 模型与分词器（Model and tokenizer）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--model-path`, `--model` | 模型权重路径，可以是本地目录或 Hugging Face 仓库 ID | None | — |
| `--tokenizer-path` | 分词器路径 | None | — |
| `--tokenizer-mode` | 分词器模式。auto：可用时使用快速分词器；slow：始终使用慢速分词器 | auto | auto / slow |
| `--tokenizer-backend` | 分词器后端。huggingface 使用默认 HuggingFace tokenizers 库；fastokens 使用 fastokens 库获得更快分词速度（需安装 fastokens 包） | huggingface | huggingface / fastokens |
| `--tokenizer-worker-num` | 分词器管理器的工作进程数 | 1 | — |
| `--skip-tokenizer-init` | 跳过分词器初始化，在生成请求中直接传入 input_ids | False | — |
| `--load-format` | 模型权重加载格式。auto 优先 safetensors、不可用时回退 pytorch bin；pt 为 pytorch bin 格式；safetensors 为 safetensors 格式；instanttensor 使用 InstantTensor 高性能分布式加载器加载 Safetensors 权重；npcache 以 pytorch 格式加载并缓存 numpy 以加速；dummy 用随机值初始化权重（主要用于性能分析）；gguf 加载 GGUF 格式；bitsandbytes 使用 bitsandbytes 量化加载；layered 逐层加载（可先量化一层再加载下一层，降低峰值内存）；flash_rl 加载 flash_rl 格式；另支持 fastsafetensors 与 private；runai_streamer 支持从对象存储和共享文件系统直接加载 | auto | auto / pt / safetensors / instanttensor / npcache / dummy / sharded_state / gguf / bitsandbytes / mistral / layered / flash_rl / remote / remote_instance / fastsafetensors / private / runai_streamer |
| `--model-loader-extra-config` | 模型加载器的额外配置，将传递给所选 load_format 对应的加载器 | {} | — |
| `--trust-remote-code` | 允许执行模型仓库中自定义建模文件里的代码 | False | — |
| `--context-length` | 模型最大上下文长度；默认 None（使用模型 config.json 中的值） | None | — |
| `--is-embedding` | 将 CausalLM 用作嵌入模型 | False | — |
| `--enable-multimodal` | 为服务模型启用多模态功能；若模型本身不是多模态则无效果 | None | — |
| `--revision` | 使用的模型版本，可以是分支名、标签名或 commit id；未指定则使用默认版本 | None | — |
| `--model-impl` | 模型实现方式。auto：存在 SGLang 实现则优先使用，否则回退 Transformers 实现；sglang：使用 SGLang 模型实现；transformers：使用 Transformers 模型实现 | auto | — |
| `--detokenizer-worker-num` | 反分词器（detokenizer）管理器的工作进程数 | 1 | — |
| `--prefill-only-disable-kv-cache` | 对嵌入模式的纯预填充负载跳过物理 KV缓存分配。目前仅在同时使用 --is-embedding、--chunked-prefill-size=-1、--disable-radix-cache、FA 预填充后端且非 FP4 KV缓存时有效（fa_skip_kv_cache 路径激活，无任何层读写缓存）。调度器准入统计不变；每层 K/V 张量按 (page_size, head_num, head_dim) 占位，不浪费 GPU 显存 | False | — |
| `--model-config-parser` | 模型配置解析器。auto 通过 is_mistral_model 名称启发式选择 mistral，否则使用 hf（基于 config.json 的 AutoConfig）；插件可通过 @register_model_config_parser 注册额外解析器 | auto | — |

### 3.2 HTTP 服务器（HTTP server）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--host` | HTTP 服务器主机地址 | 127.0.0.1 | — |
| `--port` | HTTP 服务器端口 | 30000 | — |
| `--fastapi-root-path` | 应用部署在基于路径路由的反向代理之后时设置 | "" | — |
| `--grpc-mode` | 使用 gRPC 服务器替代 HTTP 服务器 | False | — |
| `--skip-server-warmup` | 跳过服务器预热 | False | — |
| `--warmups` | 指定服务器启动前运行的自定义预热函数（逗号分隔），如 --warmups=warmup_name1,warmup_name2 会在监听请求前运行 warmup.py 中指定的函数 | None | — |
| `--nccl-port` | NCCL 分布式环境搭建端口，默认随机端口 | None | — |
| `--checkpoint-engine-wait-weights-before-ready` | 设置后，服务器等待通过 checkpoint-engine 或其他更新方法加载初始权重后再服务推理请求 | False | — |
| `--ssl-keyfile` | SSL 密钥文件路径 | None | — |
| `--ssl-certfile` | SSL 证书文件路径 | None | — |
| `--ssl-ca-certs` | CA 证书文件 | None | — |
| `--ssl-keyfile-password` | SSL 密钥文件的解密密码 | None | — |
| `--enable-ssl-refresh` | 证书/密钥文件在磁盘上变更时启用自动热重载，需同时指定 --ssl-certfile 与 --ssl-keyfile | False | — |
| `--enable-http2` | 使用 Granian 替代 Uvicorn 作为 ASGI 服务器，启用 HTTP/1.1 与 HTTP/2 自动协商；客户端可使用 h2c（明文 HTTP/2）或 HTTP/1.1；需 pip install sglang[http2] | False | — |
| `--http2-max-concurrent-streams` | 每个 HTTP/2 连接广播的最大并发流数（1 到 2^32-1），仅在 --enable-http2 时生效 | 200 | — |

### 3.3 量化与数据类型（Quantization and data type）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--dtype` | 模型权重与激活的数据类型。auto：FP32/FP16 模型用 FP16，BF16 模型用 BF16；half 即 FP16（AWQ 量化推荐）；float16 同 half；bfloat16 兼顾精度与范围；float 为 FP32 简写；float32 为 FP32 精度 | auto | auto / half / float16 / bfloat16 / float / float32 |
| `--quantization` | 量化方法 | None | awq / fp8 / mxfp8 / gptq / marlin / gptq_marlin / awq_marlin / bitsandbytes / gguf / modelopt / modelopt_fp8 / modelopt_fp4 / nvfp4_online / modelopt_mixed / petit_nvfp4 / w8a8_int8 / w8a8_fp8 / moe_wna16 / w4afp8 / mxfp4 / auto-round / compressed-tensors / modelslim / quark / quark_int4fp8_moe / quark_mxfp4 / mlx_q4 / mlx_q8 / unquant |
| `--quantization-param-path` | 包含 KV缓存缩放因子的 JSON 文件路径；KV缓存为 FP8 时应提供，否则缩放因子默认 1.0，可能影响精度 | None | — |
| `--kv-cache-dtype` | KV缓存存储数据类型。auto 使用模型数据类型；bf16/bfloat16 为 BF16 KV缓存；fp8_e5m2 与 fp8_e4m3 需 CUDA 11.8+；nvfp4 为 NVFP4 FP4 E2M1 方案；fp4_mx_block16 为块大小 16 的 FP4 E2M1 方案，两者均需 CUDA 12.8+ 与 PyTorch 2.8.0+ | auto | auto / fp8_e5m2 / fp8_e4m3 / bf16 / bfloat16 / nvfp4 / fp4_mx_block16 |
| `--enable-fp32-lm-head` | LM 头输出（logits）使用 FP32 | False | — |
| `--modelopt-quant` | ModelOpt 量化配置，支持 fp8、int4_awq、w4a8_awq、nvfp4、nvfp4_awq；需安装 NVIDIA Model Optimizer（pip install nvidia-modelopt） | None | — |
| `--modelopt-checkpoint-restore-path` | 恢复已保存 ModelOpt 量化检查点的路径；提供时跳过量化过程直接从该检查点加载 | None | — |
| `--modelopt-checkpoint-save-path` | 量化后保存 ModelOpt 量化检查点的路径，便于后续运行复用 | None | — |
| `--modelopt-export-path` | ModelOpt 量化后以 HuggingFace 格式导出模型的路径，导出后可直接用于 SGLang 推理；不提供则不导出 | None | — |
| `--quantize-and-serve` | 用 ModelOpt 量化后立即提供服务而不导出，适合开发与原型验证；生产环境建议量化与部署分步进行 | False | — |
| `--rl-quant-profile` | FlashRL 量化 profile 路径，使用 --load-format flash_rl 时必需 | None | — |
| `--enable-quant-communications` | 启用 TP 通信的 INT8 量化（仅支持 NPU 上的 Qwen3 系列） | False | — |
| `--enable-tf32-matmul` | FP32 矩阵乘法使用 TF32 精度提升性能（torch.set_float32_matmul_precision）；仅 CUDA；MiniMax-M2 与 GLM-4 模型自动启用 | False | — |

### 3.4 内存与调度（Memory and scheduling）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--mem-fraction-static` | 静态分配内存（模型权重与 KV缓存池）占比；出现 OOM 可调小。未设置时按（GPU 显存 − 预留内存）/ GPU 显存计算，无法检测时默认 0.88 | None | — |
| `--max-running-requests` | 最大运行请求数 | None | — |
| `--max-queued-requests` | 最大排队请求数；使用 disaggregation-mode 时忽略 | None | — |
| `--max-total-tokens` | 内存池最大 token 数；未指定时按内存使用比例自动计算，通常用于开发与调试 | None | — |
| `--chunked-prefill-size` | 分块预填充单块最大 token 数；设为 -1 表示禁用分块预填充 | None | — |
| `--prefill-max-requests` | 预填充批次最大请求数；未指定则不限制 | None | — |
| `--enable-dynamic-chunking` | 为流水线并行启用动态分块：按拟合函数动态计算块大小，使各块执行时间保持一致 | False | — |
| `--max-prefill-tokens` | 预填充批次最大 token 数；实际上限为该值与模型最大上下文长度中的较大者 | 16384 | — |
| `--schedule-policy` | 请求调度策略 | fcfs | lpm / random / fcfs / dfs-weight / lof / priority / routing-key |
| `--enable-priority-scheduling` | 启用优先级调度；默认优先级整数值大的请求先被调度 | False | — |
| `--abort-on-priority-when-disabled` | 优先级调度禁用时，中止带优先级字段的请求 | False | — |
| `--schedule-low-priority-values-first` | 配合 --enable-priority-scheduling 使用时，优先级整数值小的请求先被调度 | False | — |
| `--priority-scheduling-preemption-threshold` | 新请求抢占运行中请求所需的最小优先级差 | 10 | — |
| `--schedule-conservativeness` | 调度策略保守程度，值越大越保守；请求被频繁回退（retract）时增大该值 | 1.0 | — |
| `--retraction-policy` | KV缓存满时的解码回退策略。length 保持原行为（先回退短输出、长输入的请求）；priority 先回退低优先级请求（方向与优先级调度一致） | length | length / priority |
| `--page-size` | 每页 token 数 | 1 | — |
| `--enable-page-major-kv-layout` | 将 Mamba 状态与全注意力/SWA KV缓存以每 token 槽位一条目（含所有层状态）的形式存入单一页粒度缓冲区，而非每层一个缓冲区；由 --enable-unified-memory 隐含启用，目前仅在其之下支持 | False | — |
| `--enable-unified-memory` | 针对混合 Mamba/GDN 与混合 SWA 模型，用单一字节缓冲区替代静态划分的内存池（全注意力 KV + SWA/Mamba conv/SSM 状态），子池间容量随负载动态分配而非启动时固定；隐含 --enable-page-major-kv-layout；需 stride-aware 注意力、线性注意力与 Mamba 后端；允许的后端及支持的 PD 分离、投机解码配置在启动时校验 | False | — |
| `--swa-full-tokens-ratio` | SWA 层 KV token 与全注意力层 KV token 之比（与 swa:full 层数无关），取值 0 到 1；如 0.5 表示每个 SWA 层 50 token 时每个全注意力层 100 token | 0.8 | — |
| `--disable-hybrid-swa-memory` | 禁用混合 SWA 内存 | False | — |
| `--radix-eviction-policy` | 基数树缓存淘汰策略：lru 最近最少使用；lfu 最不常用 | lru | lru / lfu / slru / priority |
| `--enable-prefill-delayer` | 为 DP attention 启用预填充延迟器以减少空闲时间 | False | — |
| `--prefill-delayer-max-delay-passes` | 延迟预填充的最大前向步数 | 30 | — |
| `--prefill-delayer-token-usage-low-watermark` | 预填充延迟器的 token 使用率低水位 | None | — |
| `--prefill-delayer-queue-min-ratio` | 启用自适应队列延迟触发（独立于槽位触发）：等待队列达到 min(运行请求数 × ratio, max_prefill_bs) 前推迟预填充，使小碎片合并成更大预填充批次；未设置则保持仅槽位触发行为；典型值 0.1–0.5 | None | — |
| `--prefill-delayer-max-delay-ms` | 单次队列触发延迟的墙上时钟上限（毫秒），超时强制放行以约束最坏 TTFT；仅在设置 --prefill-delayer-queue-min-ratio 时生效；典型值 1000–5000 | None | — |
| `--prefill-delayer-forward-passes-buckets` | 预填充延迟器前向步数直方图的自定义分桶；0 与 max_delay_passes-1 自动加入 | None | — |
| `--prefill-delayer-wait-seconds-buckets` | 预填充延迟器等待秒数直方图的自定义分桶；0 自动加入 | None | — |
| `--disable-priority-preemption` | 禁用优先级调度抢占 | False | — |
| `--default-priority-value` | 未显式指定优先级的请求的默认优先级 | None | — |

### 3.5 运行时选项（Runtime options）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--device` | 运行设备（cuda、xpu、hpu、npu、cpu）；未指定时自动检测 | None | — |
| `--tensor-parallel-size`, `--tp-size` | 张量并行度 | 1 | — |
| `--pipeline-parallel-size`, `--pp-size` | 流水线并行度 | 1 | — |
| `--attention-context-parallel-size`, `--attn-cp-size` | 注意力上下文并行度 | 1 | — |
| `--dcp-size`, `--decode-context-parallel-size` | 解码上下文并行度，参见 Decode Context Parallelism 文档 | 1 | — |
| `--dcp-comm-backend` | DCP 注意力归约的通信后端：AllGather+ReduceScatter、一次 All-to-All 交换或 FlashInfer MNNVL All-to-All；未设置时，DCP 组在 Blackwell 上共享 MNNVL 域则选 fi_a2a，CUDA/ROCm/XPU 上选 a2a，否则 ag_rs | None | ag_rs / a2a / fi_a2a |
| `--dcp-replicate-q-proj` | a2a/fi_a2a 后端的 MLA DCP：复制 Q 投影使每个 DCP rank 本地计算全头 query；用 --no-dcp-replicate-q-proj 关闭模型特定的默认值 | None | — |
| `--moe-data-parallel-size`, `--moe-dp-size` | MoE 数据并行度 | 1 | — |
| `--pp-max-micro-batch-size` | 流水线并行的最大微批大小 | None | — |
| `--pp-async-batch-depth` | 流水线并行的异步批次深度 | 0 | — |
| `--stream-interval` | 流式输出的间隔（以 token 数计的缓冲大小）；值小更平滑，值大吞吐更高 | 1 | — |
| `--incremental-streaming-output` | 以不重叠片段序列的形式输出 | False | — |
| `--random-seed` | 随机种子 | None | — |
| `--mlx-enable-sampling` | 仅 MLX 后端：解码 token 采样（temperature/top-k/top-p/min-p）而非贪心 argmax；采样在惰性 MLX 图内运行，兼容 overlap 调度器；prefill/extend 的首批 token 也会采样；贪心请求保持精确 argmax；同时在 MLX 路径启用：语法词表掩码与自定义 logit 处理器（仅纯解码步运行）、logit_bias、输出 logprobs（prompt 输入 logprobs 不计算）、NaN 清理（SGLANG_SANITIZE_NAN_LOGITS）、--enable-deterministic-inference 下的 per-request sampling_seed（仅 MLX 内确定）；不应用惩罚 | False | — |
| `--constrained-json-whitespace-pattern` | （仅 outlines 与 llguidance 后端）JSON 约束输出中允许的语法空白字符正则，如允许连续空白可设为 `[\n\t ]*` | None | — |
| `--constrained-json-disable-any-whitespace` | （仅 xgrammar 与 llguidance 后端）JSON 约束输出强制紧凑表示 | False | — |
| `--watchdog-timeout` | 看门狗超时（秒）；前向批次超过该时长服务器将崩溃以防挂起 | 300 | — |
| `--soft-watchdog-timeout` | 软看门狗超时（秒）；前向批次超过该时长服务器输出调试信息 | None | — |
| `--dist-timeout` | torch.distributed 初始化超时 | None | — |
| `--download-dir` | HuggingFace 模型下载目录 | None | — |
| `--model-checksum` | 模型文件完整性校验；不带值时以 model-path 作为 HF repo ID，否则提供校验和 JSON 文件路径或 HuggingFace repo ID | None | — |
| `--base-gpu-id` | GPU 分配的起始 ID；同机运行多实例时有用 | 0 | — |
| `--gpu-id-step` | 相邻所用 GPU ID 的间隔，如设为 2 则使用 GPU 0,2,4,... | 1 | — |
| `--sleep-on-idle` | 空闲时降低 CPU 占用 | False | — |
| `--batch-notify-size` | 让出事件循环前批量处理的流式通知数，降低高并发下 asyncio 唤醒开销 | 16 | — |
| `--stream-response-default-include-usage` | 每个流式响应都包含 usage（即使未指定 stream_options） | False | — |
| `--enable-streaming-session` | 启用流式会话模式与 StreamingSession 包装器 | False | — |
| `--load-snapshot-publish-interval` | 每 N 个解码迭代向共享内存发布负载快照；预填充与空闲始终立即发布 | 15 | — |
| `--use-ray` | 使用 Ray actor 管理调度器进程 | False | — |

### 3.6 日志（Logging）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--log-level` | 所有日志器的日志级别 | info | — |
| `--log-level-http` | HTTP 服务器的日志级别；未设置时默认复用 --log-level | None | — |
| `--log-requests` | 记录所有请求的元数据、输入、输出；详细程度由 --log-requests-level 决定 | False | — |
| `--log-requests-level` | 0：仅记录元数据（不含采样参数）；1：元数据+采样参数；2：元数据+采样参数+部分输入输出；3：记录全部输入输出 | 2 | 0 / 1 / 2 / 3 |
| `--log-requests-format` | 请求日志格式：text（人类可读）或 json（结构化） | text | text / json |
| `--log-requests-target` | 请求日志输出目标：stdout 和/或文件目录路径，可指定多目标，如 --log-requests-target stdout /my/path | None | — |
| `--uvicorn-access-log-exclude-prefixes` | 排除以这些前缀开头的 uvicorn 访问日志；默认空（禁用） | [] | — |
| `--crash-dump-folder` | 崩溃诊断目录；存储 crash-dump 缓冲保留的已完成请求与进行中请求，NVIDIA CUDA 下还配置该目录下的设备 coredump；已有 CUDA coredump 环境变量优先；未指定时不启用请求 crash dump 也不设置 CUDA coredump 默认值 | None | — |
| `--show-time-cost` | 显示自定义标记的时间开销 | False | — |
| `--enable-metrics` | 启用 Prometheus 指标日志 | False | — |
| `--enable-mfu-metrics` | 启用估算 MFU 相关的 Prometheus 指标 | False | — |
| `--enable-metrics-for-all-schedulers` | 让所有 TP rank（而非仅 TP 0）的调度器分别记录请求指标；启用 dp_attention 时尤其有用，否则所有指标看似都来自 TP 0 | False | — |
| `--tokenizer-metrics-custom-labels-header` | 传递分词器指标自定义标签的 HTTP 头 | x-custom-labels | — |
| `--tokenizer-metrics-allowed-custom-labels` | 分词器指标允许的自定义标签；通过 HTTP 请求中 --tokenizer-metrics-custom-labels-header 字段以字典指定，如设置 'label1 label2' 后允许 {'label1': 'value1', 'label2': 'value2'} | None | — |
| `--bucket-time-to-first-token` | 首 token 时间（TTFT）的分桶，浮点数列表 | None | — |
| `--bucket-inter-token-latency` | token 间延迟的分桶，浮点数列表 | None | — |
| `--bucket-e2e-request-latency` | 端到端请求延迟的分桶，浮点数列表 | None | — |
| `--prompt-tokens-buckets` | prompt token 直方图分桶规则，支持 3 种：default 用预定义分桶；'tse <middle> <base> <count>' 生成两侧指数分布分桶（如 'tse 1000 2 8' 生成 [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]）；'custom <value1> <value2> ...' 用自定义分桶值（如 'custom 10 50 100 500'） | None | — |
| `--generation-tokens-buckets` | 生成 token 直方图分桶规则，支持 3 种：default 用预定义分桶；'tse <middle> <base> <count>' 生成两侧指数分布分桶；'custom <value1> <value2> ...' 用自定义分桶值 | None | — |
| `--gc-warning-threshold-secs` | 长 GC 告警阈值，GC 超过该时长记录警告；设为 0 禁用 | 0.0 | — |
| `--decode-log-interval` | 解码批次的日志间隔 | 40 | — |
| `--enable-request-time-stats-logging` | 启用每请求时间统计日志 | False | — |
| `--kv-events-config` | NVIDIA dynamo KV 事件发布的 JSON 配置；使用该 flag 即启用发布。面向负载感知路由的运行时负载发布需单独开启，见 --load-publish-endpoint；设置 replay_endpoint（如 tcp://*:5558）可让路由器重新拉取被丢弃的批次；/server_info 会公布其端口 | None | — |
| `--load-publish-endpoint` | 开启供负载感知路由订阅的运行时负载 PUB 套接字；默认关闭（unset 或 off）；auto 表示在 --kv-events-config 端口范围之后紧排 dp_size 个端口，或指定通配主机 TCP 地址（如 tcp://*:6000）显式放置，rank r 绑定 port+r，/server_info 在 kv_events 块下公布基地址；需 --kv-events-config，未设置、不可绑定或与 KV 范围重叠则启动失败；auto 从 KV 基地址预留 2*dp_size 个端口；面向路由器的更新频率跟随 --load-snapshot-publish-interval | None | — |
| `--enable-trace` | 启用 OpenTelemetry 链路追踪 | False | — |
| `--trace-modules` | 选择要追踪的组件：request、mooncake；格式：<module1 name>,<module2 name>,... | request | — |
| `--otlp-traces-endpoint` | 设置 --enable-trace 时的 OpenTelemetry collector 端点；格式：<ip>:<port> | localhost:4317 | — |
| `--grpc-http-sidecar-port` | gRPC 模式（--grpc-mode）下 HTTP 边车服务器端口，提供 Prometheus 指标与 profiling 端点；默认 --port + 1；HTTP 模式不使用 | None | — |
| `--extra-metric-labels` | 指标自定义标签，如 '{"label1": "value1", "label2": "value2"}' | None | — |
| `--enable-forward-pass-metrics` | 通过 ZMQ IPC 启用每迭代前向 pass 指标；外部消费者（如 Dynamo planner）订阅 server_args.forward_pass_metrics_ipc_name 暴露的 IPC 端点 | False | — |
| `--forward-pass-metrics-worker-id` | — | "" | — |
| `--forward-pass-metrics-ipc-name` | — | None | — |

### 3.7 请求指标导出器配置（RequestMetricsExporter configuration）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--export-metrics-to-file` | 将每请求的性能指标导出到本地文件（如转发给外部系统） | False | — |
| `--export-metrics-to-file-dir` | 性能指标文件的写入目录（启用 --export-metrics-to-file 时必需） | None | — |

### 3.8 响应存储（Responses storage）

响应存储（含独立服务器）默认禁用，需启用 `--enable-response-store` 才能保留响应供检索、previous_response_id 链式调用与后台请求使用。存储为进程本地内存，无 TTL 与大小限制，重启即丢失；后台请求（含流式）需同时设置 `store: true`；该参数与 `--disaggregation-mode=prefill` 或 `decode` 组合会在启动时报错，PD 客户端须在前台请求中显式发送完整对话历史。

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--enable-response-store` | 启用响应存储，用于响应检索、previous_response_id 链式调用与后台请求；存储为进程本地内存，无 TTL 与大小限制，重启丢失；后台请求需同时设置 store: true；与 PD 分离模式（--disaggregation-mode=prefill/decode）不兼容 | False | — |

### 3.9 API 相关（API related）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--api-key` | 服务器 API 密钥，也用于 OpenAI API 兼容服务器 | None | — |
| `--admin-api-key` | 管理/控制端点（权重更新、缓存刷新、/server_info 等）的管理员 API 密钥；设置后仅管理员端点要求 Authorization: Bearer <admin_api_key> | None | — |
| `--served-model-name` | 覆盖 OpenAI API 服务器 v1/models 端点返回的模型名 | None | — |
| `--weight-version` | 模型权重版本标识；未指定默认 default | default | — |
| `--chat-template` | 内置聊天模板名或聊天模板文件路径；仅用于 OpenAI 兼容 API 服务器 | None | — |
| `--hf-chat-template-name` | HuggingFace 分词器含多个命名聊天模板（如 default、tool_use、rag）时指定使用哪一个；未设置时使用第一个可用模板 | None | — |
| `--default-chat-template-kwargs` | 作为 chat_template_kwargs 默认值的 JSON 对象，作用于 Chat Completions、Responses、Anthropic 兼容请求与聊天 token 化；直接 Ollama 与嵌入路径不消费这些默认值；普通模板键上显式的每请求值优先；键取决于所选模型的聊天模板；reasoning_effort 单独归一化 | None | — |
| `--completion-template` | 内置补全模板名或补全模板文件路径；仅用于 OpenAI 兼容 API 服务器，目前仅支持代码补全 | None | — |
| `--file-storage-path` | 后端文件存储路径 | sglang_storage | — |
| `--enable-cache-report` | 在每个 OpenAI 请求的 usage.prompt_tokens_details 中返回缓存 token 数 | False | — |
| `--reasoning-parser` | 推理模型的思考内容解析器；auto 表示从模型聊天模板自动检测 | None | auto / apertus2509 / deepseek-r1 / deepseek-v3 / deepseek-v4 / dots / glm45 / ling3 / hunyuan / gpt-oss / k2_horizon / kimi / kimi_k2 / kimi_k3 / mimo / muse / poolside_v1 / qwen3 / qwen3-thinking / minimax / minimax-append-think / minimax-m3 / step3 / step3p5 / mistral / nemotron_3 / interns1 / gemma4 / gigachat35 / inkling / cohere_command4 |
| `--tool-call-parser` | 工具调用交互的解析器；auto 表示从模型聊天模板自动检测 | None | auto / apertus2509 / cohere_command4 / deepseekv3 / deepseekv31 / deepseekv32 / deepseekv4 / dots / glm / glm45 / glm47 / gpt-oss / k2_horizon / kimi_k2 / kimi_k3 / lfm2 / ling3 / llama3 / mimo / minicpm5 / mistral / muse / poolside_v1 / pythonic / qwen / qwen25 / qwen3_coder / spark25 / step3 / step3p5 / minimax-m2 / minimax-m3 / trinity / interns1 / hermes / hunyuan / gigachat3 / gigachat35 / gemma4 / inkling |
| `--tool-server` | demo 或逗号分隔的工具服务器 URL 列表；未指定则不使用外部工具服务器；原生 GPT-OSS web_search 仍可通过 EXA_API_KEY 启用 | None | — |
| `--sampling-defaults` | 默认采样参数来源：openai 用 SGLang/OpenAI 默认值（temperature=1.0、top_p=1.0 等）；model 用模型 generation_config.json 中的推荐参数 | model | openai / model |
| `--strip-thinking-cache` | 结束时跳过将推理模型输出（思考+回答）缓存进基数树，仅保留 prompt 前缀；为可选功能，会改变缓存内容 | False | — |
| `--enable-strict-thinking` | 思考阶段启用严格 token 过滤，阻止模型特有的排除 token（如工具调用标记）；需支持 token 过滤的语法后端；每请求思考上限见 Thinking Budget | False | — |
| `--asr-max-buffer-seconds` | 流式 ASR WebSocket 处理器在以 buffer_overflow 错误关闭会话前累积的 PCM 音频最大秒数；防止客户端音频流快于推理消费导致 OOM | 60 | — |
| `--asr-max-concurrent-sessions` | /v1/realtime 服务的实时 ASR WebSocket 最大并发会话数；超出的新连接被接受后发送 error{code:too_many_sessions} 帧并关闭 | 32 | — |

### 3.10 数据并行（Data parallelism）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--data-parallel-size`, `--dp-size` | 模型数据并行副本数 | 1 | — |
| `--attention-data-parallel-size`, `--attn-dp-size` | 注意力数据并行度：TP 组内数据并行的注意力组数（每组各一个调度器），FFN 保持张量并行 | 1 | — |
| `--load-balance-method` | 数据并行的负载均衡策略；total_tokens 算法仅可用于 DP attention，基于 DP worker 的实时 token 负载进行均衡 | auto | auto / round_robin / follow_bootstrap_room / total_requests / total_tokens |

### 3.11 多节点分布式服务（Multi-node distributed serving）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--dist-init-addr`, `--nccl-init-addr` | 分布式后端初始化的主机地址（如 192.168.0.2:25000） | None | — |
| `--nnodes` | 节点数 | 1 | — |
| `--node-rank` | 节点 rank | 0 | — |

### 3.12 模型覆盖参数（Model override args）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--json-model-override-args` | JSON 字符串格式的字典，用于覆盖模型默认配置 | {} | — |
| `--preferred-sampling-params` | JSON 格式的采样设置，将在 /get_model_info 中返回 | None | — |

### 3.13 LoRA（LoRA）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--enable-lora` | 启用模型 LoRA 支持；为向后兼容，提供 --lora-paths 时自动置 True | False | — |
| `--enable-lora-overlap-loading` | 启用异步 LoRA 权重加载，使 H2D 传输与 GPU 计算重叠；LoRA 负载受适配器权重加载瓶颈时（如频繁加载大 LoRA 适配器）应启用 | False | — |
| `--max-lora-rank` | 支持的最大 LoRA rank；未指定时从 --lora-paths 提供的适配器自动推断；预期启动后动态加载更大 rank 适配器时需设置 | None | — |
| `--lora-target-modules` | 应用 LoRA 的目标模块并集（如 q_proj、k_proj、gate_proj）；未指定时从 --lora-paths 适配器自动推断；设为 all 对全部支持模块启用（可能带来轻微性能开销） | None | q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj / qkv_proj / gate_up_proj / all |
| `--lora-paths` | 待加载的 LoRA 适配器列表，每个适配器格式：<PATH>、<NAME>=<PATH> 或 JSON（schema 为 {"lora_name": str, "lora_path": str, "pinned": bool}） | None | — |
| `--max-loras-per-batch` | 运行批次（含纯 base 请求）的最大适配器数 | 8 | — |
| `--max-loaded-loras` | 限制同时在 CPU 内存中加载的 LoRA 适配器数；须 ≥ --max-loras-per-batch | None | — |
| `--lora-eviction-policy` | GPU 内存池满时的 LoRA 适配器淘汰策略 | lru | lru / fifo |
| `--lora-backend` | 多 LoRA 服务的内核后端 | csgmv | triton / csgmv / ascend / torch_native |
| `--max-lora-chunk-size` | ChunkedSGMV LoRA 后端的最大块大小；仅 --lora-backend 为 csgmv 时使用；更大的值可能提升性能 | 16 | 16 / 32 / 64 / 128 |
| `--lora-drain-wait-threshold` | 任一 LoRA 适配器请求等待超过该阈值（秒）时，调度器选择性排空一个运行中的适配器腾出空间，防止少量适配器垄断批次槽位，缓解高负载或偏斜负载下的极端尾延迟；设为 0 禁用排空 | 0.0 | — |
| `--experts-shared-outer-loras` | 对 MoE 模型强制共享外层 LoRA 模式：w1/w3 的 lora_A 与 w2 的 lora_B 在专家间共享（expert_dim=1）；用 --no-experts-shared-outer-loras 强制禁用；默认从适配器权重自动检测 | None | — |
| `--lora-use-virtual-experts` | 对 MoE 模型启用虚拟专家计算 | False | — |
| `--lora-strict-loading` | 启用 LoRA 适配器严格加载：适配器权重中不匹配或缺失的键将报错 | False | — |

### 3.14 内核后端：注意力、采样、语法、GEMM（Kernel Backends (Attention, Sampling, Grammar, GEMM)）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--attention-backend` | 注意力层内核选择 | None | triton / torch_native / flex_attention / dsa / nsa / dsv4 / compressed / fa3 / fa4 / flashinfer / flashmla / trtllm_mla / cutedsl_mla / tokenspeed_mla / trtllm_mha / aiter / wave / intel_amx / ascend / intel_xpu |
| `--prefill-attention-backend` | 预填充注意力层内核（优先于 --attention-backend） | None | triton / torch_native / flex_attention / dsa / nsa / dsv4 / compressed / fa3 / fa4 / flashinfer / flashmla / trtllm_mla / cutedsl_mla / tokenspeed_mla / trtllm_mha / aiter / wave / intel_amx / ascend / intel_xpu |
| `--decode-attention-backend` | 解码注意力层内核（优先于 --attention-backend） | None | triton / torch_native / flex_attention / dsa / nsa / dsv4 / compressed / fa3 / fa4 / flashinfer / flashmla / trtllm_mla / cutedsl_mla / tokenspeed_mla / trtllm_mha / aiter / wave / intel_amx / ascend / intel_xpu |
| `--sampling-backend` | 采样层内核选择 | None | flashinfer / pytorch / ascend |
| `--grammar-backend` | 语法引导解码的后端 | None | xgrammar / outlines / llguidance / none |
| `--mm-attention-backend` | 多模态注意力后端 | None | sdpa / fa3 / fa4 / triton_attn / ascend_attn / aiter_attn / flashinfer_cudnn / amx_attn / xpu_attn |
| `--dsa-prefill-backend` | 预填充阶段的 DSA 后端（运行 DeepSeek DSA 风格注意力时覆盖 --attention-backend） | 自动（取决于硬件） | flashmla_sparse / flashmla_sparse_q8 / triton_sparse_mla / flashmla_kv / flashmla_auto / fa3 / tilelang / aiter / trtllm |
| `--dsa-triton-union` | --dsa-prefill-backend 为 triton_sparse_mla 时的 union 组大小：相邻 query token 共享一个收集索引集（精确；所有权掩码恢复各 token 自己的 softmax）；要求 num_q_heads * union 为 [16, 32] 内 2 的幂；--enable-deterministic-inference 下强制关闭 | 0 | 0 / 2 / 4 |
| `--dsa-decode-backend` | 解码阶段的 DSA 后端（运行 DeepSeek DSA 风格注意力时覆盖 --attention-backend） | 自动（取决于硬件） | flashmla_sparse / flashmla_kv / flashmla_auto / fa3 / tilelang / aiter / trtllm |
| `--dsa-topk-backend` | 目标模型的 DSA 索引器 top-k 后端；torch 后端目前要求 SGLANG_DSA_FUSE_TOPK=false | sgl-kernel | sgl-kernel / torch / flashinfer |
| `--enable-deepseek-v4-fp4-indexer` | 启用 DeepSeek V4 实验性 FP4 C4 索引器路径；未设置时保持默认 DeepSeek V4 索引器路径；需支持 DeepGEMM FP4 索引器的 SM100 GPU | False | — |
| `--fp8-gemm-backend` | Blockwise FP8 GEMM 的运行后端。auto：MXFP8 密集 GEMM 在 FlashInfer 报告支持时（目前 SM100/SM103）选 flashinfer_cutedsl，否则在支持的 Blackwell GPU 上选 flashinfer_cutlass；另有 deep_gemm（JIT 编译）、flashinfer_trtllm（仅 SM100/SM103）、flashinfer_cutlass、flashinfer_cutedsl（仅 SM100/SM103）、flashinfer_deepgemm（仅 Hopper SM90，解码小 M 维用 swapAB 优化）、cutlass（SM120 GPU 最优）、triton（兜底，兼容性广）、aiter（仅 ROCm） | auto | auto / deep_gemm / flashinfer_trtllm / flashinfer_cutlass / flashinfer_cutedsl / flashinfer_deepgemm / cutlass / triton / aiter |
| `--fp4-gemm-backend` | NVFP4 GEMM 的运行后端。auto：SM100 选 flashinfer_cutedsl，SM80-SM90 选 marlin，其余（含 SM120）选 flashinfer_cutlass；flashinfer_cutlass 为 FlashInfer CUTLASS 后端；flashinfer_cudnn 在 CUDA 13+ 与 cuDNN 9.15+ 最优；flashinfer_cutedsl 为 FlashInfer CuTe DSL 后端；flashinfer_trtllm 需不同的权重准备（含重排）；marlin 为 SM80-SM90 的 W4A16 兜底；需安装 FlashInfer | auto | auto / flashinfer_cudnn / flashinfer_cutedsl / flashinfer_cutlass / flashinfer_trtllm / marlin |
| `--bf16-gemm-backend` | 未量化 BF16 GEMM 的后端。auto：SM10x GPU 选 cutedsl（确定性推理时选 torch），否则经 torch.nn.functional.linear 用 cuBLAS；cutedsl：SM10x 上 SGLang JIT CuTe DSL TGV BF16 GEMM（在 CuTe DSL 内核与 cuBLAS 间分派）；torch：始终经 torch.nn.functional.linear 用 cuBLAS | auto | auto / cutedsl / torch |
| `--disable-flashinfer-autotune` | FlashInfer 自动调优默认开启；设置该 flag 禁用自动调优 | False | — |
| `--flashinfer-autotune-skip-ops` | 自动调优时跳过的 FlashInfer 自定义算子标识，详见 FlashInfer 自动调优文档；被跳过的算子用启发式兜底；因 IMA 问题 SGLang 默认临时跳过 mxfp8_gemm | None | — |
| `--radix-cache-backend` | 经 register_radix_cache_backend 注册的基数树缓存后端名；省略该 flag 则使用内置默认缓存选择链 | None | — |

### 3.15 投机解码（Speculative decoding）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--speculative-algorithm` | 投机解码算法 | None | EAGLE / EAGLE3 / NEXTN / STANDALONE / NGRAM |
| `--speculative-draft-model-path`, `--speculative-draft-model` | draft 模型权重路径，可以是本地目录或 Hugging Face 仓库 ID | None | — |
| `speculative_boundary_reduction`（服务器参数字段，无 CLI flag） | 投机 draft 模型的 FFN 边界归约策略；参数处理会从 boundary_reduction 与 draft 模型配置推导该字段，独立设置的值会被覆盖；参见 FFN boundary reduction | 解析前为 rs+rsv；之后取决于模型 | ar / rs / rsv / rs+rsv |
| `--speculative-draft-model-revision` | draft 模型版本，可以是分支名、标签名或 commit id；未指定则使用默认版本 | None | — |
| `--speculative-draft-load-format` | draft 模型权重加载格式；未指定时与 --load-format 相同；用 dummy 可将 draft 权重以随机值初始化（性能分析用） | None | auto / pt / safetensors / instanttensor / npcache / dummy / sharded_state / gguf / bitsandbytes / mistral / layered / flash_rl / remote / remote_instance / fastsafetensors / private / runai_streamer |
| `--speculative-num-steps` | 投机解码中从 draft 模型采样的步数 | None | — |
| `--speculative-eagle-topk` | eagle2 中每步从 draft 模型采样的 token 数 | None | — |
| `--speculative-num-draft-tokens` | 投机解码中从 draft 模型采样的 token 数 | None | — |
| `--speculative-accept-threshold-single` | draft token 在目标模型中的概率大于该阈值时接受 | 1.0 | — |
| `--speculative-accept-threshold-acc` | draft token 的接受概率从目标概率 p 提升到 min(1, p / threshold_acc) | 1.0 | — |
| `--speculative-use-block-verification` | CUDA 或 ROCm 上 EAGLE/EAGLE3/NEXTN 使用块验证（需 topk=1） | False | — |
| `--speculative-token-map` | draft 模型小词表表的路径 | None | — |
| `--speculative-attention-mode` | 投机解码操作（目标验证与 draft 扩展）的注意力后端模式：prefill（默认）或 decode | prefill | prefill / decode |
| `--speculative-draft-attention-backend` | 投机解码 draft 的注意力后端 | None | 同 --attention-backend 可选值 |
| `--speculative-dsa-topk-backend` | 投机 draft worker 的 DSA 索引器 top-k 后端；torch 后端目前要求 SGLANG_DSA_FUSE_TOPK=false | sgl-kernel | sgl-kernel / torch / flashinfer |
| `--speculative-moe-runner-backend` | EAGLE 投机解码的 MoE 后端（选项同 --moe-runner-backend）；未设置时同 MoE runner 后端 | None | auto / deep_gemm / triton / triton_kernel / flashinfer_trtllm / experimental_sgl_trtllm / flashinfer_trtllm_routed / flashinfer_cutlass / flashinfer_mxfp4 / flashinfer_cutedsl / cutlass / aiter / marlin |
| `--speculative-moe-a2a-backend` | EAGLE 投机解码的 MoE A2A 后端（选项同 --moe-a2a-backend）；未设置时同 MoE A2A 后端 | None | none / deepep / mooncake / nixl / mori / ascend_fuseep / flashinfer / megamoe / pplx |
| `--speculative-enable-w4a4-mxfp4-megamoe` | draft 模型的 MXFP4 MegaMoE 层是否使用 W4A4 mxf4xmxf4 MMA 类型；传 --no-speculative-enable-w4a4-mxfp4-megamoe 可保持 draft 为 fp8xfp4（W4A8）而目标模型运行 W4A4；未设置时同 --enable-w4a4-mxfp4-megamoe；draft 与目标 MMA 类型不同时可能在 KV 池确定后额外分配 MegaMoE 对称缓冲，CUDA 图捕获 OOM 请调低 --mem-fraction-static | None | — |
| `--speculative-draft-model-quantization` | draft 模型的量化方法 | None | 同 --quantization 可选值 |
| `--speculative-dflash-block-size` | 仅 DFLASH：块大小（验证窗口长度），即 DFLASH 的 --speculative-num-draft-tokens 别名 | None | — |
| `--speculative-draft-window-size` | draft 模型滑动窗口大小；仅 Llama EAGLE-3（LlamaForCausalLMEagle3）与 DFLASH 生效，其他 EAGLE-3 后端（如基于 MLA 的 drafter）静默忽略；Llama EAGLE-3 中 drafter 只关注最近 N 个 key（验证器隐状态+自身输出），验证器不受影响；DFLASH 中 draft worker 在本地 KV缓存中保留近期目标 token 窗口（分页后端左侧可能因对齐多保留一页）；默认为全注意力/上下文 | None | — |

### 3.16 Ngram 投机解码（Ngram speculative decoding）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--speculative-ngram-min-bfs-breadth` | ngram 投机解码中 BFS（广度优先搜索）的最小广度 | 1 | — |
| `--speculative-ngram-max-bfs-breadth` | BFS 的最大广度 | 10 | — |
| `--speculative-ngram-match-type` | ngram 树构建模式：BFS 按近期性扩展，PROB 按频率扩展；该设置传递给 ngram 缓存实现 | BFS | BFS / PROB |
| `--speculative-ngram-max-trie-depth` | ngram trie 存储与匹配的最大后缀长度 | 18 | — |
| `--speculative-ngram-capacity` | ngram 投机解码的缓存容量 | 10000000 | — |
| `--speculative-ngram-external-corpus-path` | 启动时预加载进 SAM 的外部 JSONL 语料路径；运行时可通过 POST /add_external_corpus 追加语料 | None | — |
| `--speculative-ngram-external-sam-budget` | ngram 投机解码中为外部 SAM 子树预留的 draft 节点数 | 0 | — |
| `--speculative-ngram-external-corpus-max-tokens` | 分词后的外部 ngram 语料超过该 token 数则启动失败；请按 CPU 内存预算调整 | 10000000 | — |
| `--speculative-adaptive` | 启用自适应投机解码：按接受率动态调整 num_steps | False | — |
| `--speculative-adaptive-config` | 自适应投机解码调优参数的 JSON 配置文件路径 | None | — |
| `--speculative-skip-dp-mlp-sync` | 投机解码与 DP attention 同时启用时，跳过调度器合并新批次前的额外 MLP 同步 | False | — |

### 3.17 多层 Eagle 投机解码（Multi-layer Eagle speculative decoding）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--enable-multi-layer-eagle` | 启用多层 Eagle 投机解码 | False | — |

### 3.18 MoE 混合专家（MoE）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--expert-parallel-size`, `--ep-size`, `--ep` | 专家并行度 | 1 | — |
| `--moe-a2a-backend` | 专家并行 all-to-all 通信的后端 | none | none / deepep / mooncake / nixl / mori / ascend_fuseep / flashinfer / megamoe / pplx |
| `--moe-runner-backend` | MoE 的运行后端 | auto | auto / deep_gemm / triton / triton_kernel / flashinfer_trtllm / experimental_sgl_trtllm / flashinfer_trtllm_routed / flashinfer_cutlass / flashinfer_mxfp4 / flashinfer_cutedsl / cutlass / aiter / marlin |
| `--flashinfer-mxfp4-moe-precision` | flashinfer mxfp4 MoE 的计算精度；SM90 上 fp8 选择 FlashInfer #3738 引入的 Humming 风格 MXFP4 权重 x FP8 激活路径，需 FlashInfer >= 0.6.18 | default | default / bf16 / fp8 |
| `--enable-flashinfer-allreduce-fusion` | 启用 FlashInfer allreduce 与残差 RMSNorm 融合 | False | — |
| `--enable-aiter-allreduce-fusion` | 启用 aiter allreduce 与残差 RMSNorm 融合 | False | — |
| `--deepep-mode` | 启用 DeepEP MoE 时的模式：normal、low_latency 或 auto；默认 auto，即解码批次用 low_latency、预填充批次用 normal | auto | normal / low_latency / auto |
| `--deepep-dispatcher-output-dtype` | DeepEP 分发器输出 dtype：bf16、fp8、int8（仅 Ascend A2/A3 系列 NPU）、nvfp4 或 auto；auto 按优先级顺序（服务器参数 → 已废弃环境变量 → input_global_scale 检查 → quant_config 的 dispatcher_output_dtype → flashinfer/cutlass 后端 → NPU 默认 BF16 → GPU 默认 FP8） | auto | auto / bf16 / fp8 / int8 / nvfp4 |
| `--flashinfer-a2a-dispatch-type` | FlashInfer A2A 分发器激活 dtype；未设置时回退 SGLANG_MOE_NVFP4_DISPATCH 环境变量；显式 auto 对 --quantization mxfp8 选 mxfp8、对 modelopt FP4 或混合 NVFP4 MoE 检查点选 nvfp4、其余选 bf16；该环境变量不能与此参数同用 | None | auto / bf16 / nvfp4 / mxfp8 |
| `--ep-num-redundant-experts` | 专家并行中分配的冗余专家数 | 0 | — |
| `--ep-dispatch-algorithm` | 专家并行中为冗余专家选择 rank 的算法 | None | — |
| `--init-expert-location` | EP 专家的初始位置 | trivial | — |
| `--enable-eplb` | 启用 EPLB（专家并行负载均衡）算法 | False | — |
| `--eplb-algorithm` | 所选的 EPLB 算法 | auto | — |
| `--eplb-rebalance-num-iterations` | 自动触发 EPLB 再平衡的迭代数 | 1000 | — |
| `--eplb-rebalance-layers-per-chunk` | 每次前向传播再平衡的层数 | None | — |
| `--eplb-min-rebalancing-utilization-threshold` | 触发 EPLB 再平衡的 GPU 平均利用率最小阈值；范围 [0.0, 1.0] | 1.0 | — |
| `--expert-distribution-recorder-mode` | 专家分布记录器模式 | None | — |
| `--expert-distribution-recorder-buffer-size` | 专家分布记录器环形缓冲大小；设为 -1 表示无限缓冲 | None | — |
| `--expert-balancedness-report-mode` | 专家均衡度的报告位置 | off | off / server_log / prometheus / both |
| `--deepep-config` | 适配自有集群的 DeepEP 调优配置，可以是 JSON 内容字符串或文件路径 | None | — |
| `--moe-dense-tp-size` | MoE 稠密 MLP 层的 TP 大小；大 TP 下 MLP 层权重维度小于 GEMM 支持的最小维度而报错时有用 | none | — |
| `--elastic-ep-backend` | 弹性 EP 的集合通信后端；目前支持 mooncake | None | none / mooncake / nixl |
| `--enable-elastic-expert-backup` | 启用弹性 EP 后端在 DRAM 中备份专家权重的特性；目前支持 mooncake | False | — |
| `--mooncake-ib-device` | Mooncake 后端传输使用的 InfiniBand 设备，接受逗号分隔的多个设备（如 --mooncake-ib-device mlx5_0,mlx5_1）；默认 None，启用 Mooncake 后端时自动检测 | None | — |
| `--enable-waterfill` | 启用 Waterfill：将融合共享专家作为额外路由专家槽分发到负载最小的 EP rank；支持 DeepEP 与 MegaMOE MoE A2A 后端，隐含启用共享专家融合，配 DeepEP 时支持 auto/normal/low_latency 模式（生产 DeepEP 解码建议 auto 或 low_latency 以保持 CUDA 图启用）；支持 DeepSeek-V3/R1 且 EP >= 2；默认使用静态本地批次路径，设 SGLANG_DISABLE_STATIC_WATERFILL=1 强制走运行时 EP all-reduce 的动态 Waterfill | False | — |
| `--flashinfer-allreduce-fusion-backend` | 启用 FlashInfer allreduce 融合并选择后端；默认 auto：SM90 单节点系统与 SM100/SM103 单/多节点系统选 mnnvl，否则选 trtllm；trtllm 仅单节点系统可用；mnnvl 用于 SM90 单节点与 SM100/SM103 单/多节点系统（经 MNNVL fabric）；为支持的 MoE 模型融合 allreduce 与残差+RMSNorm | None | auto / trtllm / mnnvl |
| `--enforce-disable-flashinfer-allreduce-fusion` | 强制禁用 FlashInfer allreduce 融合 | False | — |

### 3.19 Mamba 缓存（Mamba Cache）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--max-mamba-cache-size` | Mamba 缓存的最大容量 | None | — |
| `--mamba-ssm-dtype` | Mamba 缓存中 SSM 状态的数据类型；未设置时从模型配置读取 | 自动（模型配置） | float32 / bfloat16 / float16 |
| `--enable-mamba-cache-stochastic-rounding` | 写 FP16 Mamba SSM 缓存状态时启用随机舍入；需 --mamba-ssm-dtype float16 与 CUDA；--mamba-backend triton 时需 SM100 | False | — |
| `--mamba-cache-philox-rounds` | FP16 Mamba SSM 缓存写入随机舍入的 Philox 轮数；设为 0 时 Triton 用其默认值，FlashInfer 用 10 轮 | 0 | — |
| `--mamba-full-memory-ratio` | Mamba 状态内存与全 KV缓存内存之比 | 0.9 | — |
| `--mamba-radix-cache-strategy` | Mamba 调度器策略。auto 当前默认 no_buffer；no_buffer 不分配额外状态缓冲、不支持 overlap 调度器（分支点缓存可行但未实现）；extra_buffer 分配额外状态缓冲以支持 overlap 调度（非投机下每运行请求 Mamba 状态用量变为 2 倍，投机下为 1+(1/(2+speculative_num_draft_tokens)) 倍，如 speculative_num_draft_tokens==4 时 1.16 倍），非 KV缓存受限场景严格更优，分支点缓存仅 extra_buffer 支持；extra_buffer_lazy 每请求只分配一个跟踪槽降低开销（第二槽按需分配，投机解码时在每个验证窗口前预留、仅在接受的边界跨越时提交），兼容投机解码（EAGLE/NGRAM/DSPARK/DFLASH），PD 分离下不支持 | auto | auto / no_buffer / extra_buffer / extra_buffer_lazy |
| `--mamba-track-interval` | 解码期间跟踪 Mamba 状态的间隔（token 数）；仅 --mamba-radix-cache-strategy 为 extra_buffer 时使用；设置时须可被 page_size 整除，使用投机解码时须 >= speculative_num_draft_tokens | 256 | — |
| `--enable-int8-mamba-checkpoint` | 将基数树缓存的线性注意力（Mamba）状态以 int8 存储（独立检查点池），同内存下缓存前缀容量约 2 倍 | False | — |
| `--int8-mamba-ckpt-size` | int8 Mamba 检查点槽位数（默认为活动 Mamba 池大小的 2 倍） | None | — |
| `--mamba-backend` | Mamba SSM 算子的内核后端；默认 triton；flashinfer 需带 Mamba 支持的 FlashInfer | triton | triton / flashinfer |
| `--linear-attn-backend` | 线性注意力（GDN/KDA）的默认内核后端；可被 --linear-attn-decode-backend 与 --linear-attn-prefill-backend 按模式覆盖 | triton | triton / cutedsl / flashinfer |
| `--linear-attn-decode-backend` | 覆盖线性注意力解码的内核后端；未设置时用 --linear-attn-backend | None | triton / cutedsl / flashinfer |
| `--linear-attn-prefill-backend` | 覆盖线性注意力预填充/扩展的内核后端；未设置时用 --linear-attn-backend | None | triton / cutedsl / flashinfer |

### 3.20 分层缓存（Hierarchical cache）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--enable-hierarchical-cache` | 启用分层缓存 | False | — |
| `--hicache-ratio` | 主机（CPU）KV缓存内存池与设备（GPU）池大小之比 | 2.0 | — |
| `--hicache-size` | 主机 KV缓存内存池大小（GB）；设置后覆盖 hicache_ratio | 0 | — |
| `--hicache-write-policy` | 分层缓存的写策略 | write_through | write_back / write_through / write_through_selective |
| `--hicache-io-backend` | CPU 与 GPU 间 KV缓存传输的 IO 后端 | kernel | direct / kernel / kernel_ascend |
| `--hicache-mem-layout` | 分层缓存主机内存池的布局 | page_first | layer_first / page_first / page_first_direct / page_first_kv_split / page_head |
| `--hicache-storage-backend` | 分层 KV缓存的存储后端；内置：file、mooncake、hf3fs、nixl、aibrix；动态后端用 --hicache-storage-backend-extra-config 指定 backend_name（自定义名）、module_path（Python 模块路径）、class_name（后端类名） | None | file / mooncake / hf3fs / nixl / aibrix / dynamic / eic / simm |
| `--hicache-storage-prefetch-policy` | 控制何时停止从存储后端预取 | timeout | best_effort / wait_complete / timeout |
| `--hicache-storage-backend-extra-config` | JSON 字符串格式的字典、或以 @ 开头后跟 JSON/YAML/TOML 配置文件的字符串，包含存储后端的额外配置 | None | — |

### 3.21 分层稀疏注意力（Hierarchical sparse attention）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--hisparse-config`, `--hierarchical-sparse-attention-extra-config` | 分层稀疏注意力配置（JSON 字符串字典）；必填字段：algorithm（str）、backend（str）；其余字段为算法特定并传给算法构造器 | None | — |
| `--enable-hisparse` | 启用分层稀疏注意力 | False | — |

### 3.22 LMCache（LMCache）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--enable-lmcache` | 使用 LMCache 作为替代的分层缓存方案 | False | — |
| `--lmcache-config-file` | LMCache YAML 配置文件路径 | None | — |

### 3.23 Ktransformers（Ktransformers）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--kt-weight-path` | [ktransformers 参数] AMX 内核的量化专家权重路径（本地目录） | None | — |
| `--kt-method` | [ktransformers 参数] CPU 执行的量化格式 | AMXINT4 | — |
| `--kt-cpuinfer` | [ktransformers 参数] CPUInfer 线程数 | None | — |
| `--kt-threadpool-count` | [ktransformers 参数] 与 NUMA 节点数一一对应（每 NUMA 一个线程池） | 2 | — |
| `--kt-num-gpu-experts` | [ktransformers 参数] GPU 专家数 | None | — |
| `--kt-max-deferred-experts-per-token` | [ktransformers 参数] 每 token 延迟到 CPU 的最大专家数；除最后一层外的所有 MoE 层使用该值，最后一层恒为 0 | None | — |

### 3.24 扩散语言模型（Diffusion LLM）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--dllm-algorithm` | 扩散语言模型算法，如 LowConfidence | None | — |
| `--dllm-algorithm-config` | 扩散语言模型算法配置；须为 YAML 文件 | None | — |
| `--dllm-fdfo` | 先完成先出（FDFO）调度：完成的请求立即离开批次而非等待慢请求，消除队头阻塞；默认启用，传 --no-dllm-fdfo 回退同步锁步调度；兼容任意 dLLM 算法 | True | — |

### 3.25 卸载（Offloading）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--cpu-offload-gb` | 为 CPU 卸载预留的 RAM GB 数 | 0 | — |
| `--offload-group-size` | 卸载时每组的层数 | -1 | — |
| `--ple-offload-embedding` | 将 Qwen4-Exp 的 PLE n-gram 嵌入表保留在主机内存而非显存，gather 内核从主机指针读取；CUDA 上 BF16 Qwen4-Exp 默认启用，--no-ple-offload-embedding 禁用 | None | — |
| `--ple-offload-backend` | 卸载 PLE 表的主机存储：pinned（CPU 锁页内存）或 file（--ple-offload-dir 下稀疏文件支持的 mmap，由 gather 内核直接读取）；GB10 / DGX Spark 等统一内存设备用 file（其锁页主机内存与权重同池），需设备支持 cudaDevAttrPageableMemoryAccessUsesHostPageTables | pinned | — |
| `--ple-offload-dir` | --ple-offload-backend 为 file 时 PLE 表所在目录；文件为稀疏文件且每次启动重写，请放快速本地存储（NVMe）；重启前请停止所有使用该目录的服务并删除 ple_table_*.bin 文件以避免启动开销 | $SGLANG_CACHE_DIR/ple/ | — |
| `--offload-num-in-group` | 组内被卸载的层数 | 1 | — |
| `--offload-prefetch-step` | 卸载的预取步数 | 1 | — |
| `--offload-mode` | 卸载模式 | cpu | — |

### 3.26 多项评分相关参数（Args for multi-item scoring）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--enable-mis` | 启用多项评分（MIS）优化：将查询与多个 item 合并为单序列以高效批处理；需 --attention-backend flashinfer；自动禁用 CUDA 图、基数树缓存与分块预填充 | False | — |

### 3.27 优化/调试选项（Optimization/debug options）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--disable-radix-cache` | 禁用 RadixAttention 前缀缓存 | False | — |
| `--cuda-graph-config` | 按阶段的标准 CUDA 图 JSON 设置，如 {"decode":{"backend":"full","max_bs":256},"prefill":{"backend":"breakable"}}；JSON 优先于各阶段 --cuda-graph-* 便捷 flag 与旧 flag；允许的后端：full、breakable、disabled | None | — |
| `--cuda-graph-backend-decode` | 解码阶段后端；并入 cuda_graph_config[decode].backend | None | full / breakable / disabled |
| `--cuda-graph-backend-prefill` | 预填充阶段后端；并入 cuda_graph_config[prefill].backend | None | breakable / disabled |
| `--cuda-graph-max-bs-decode` | 解码 CUDA 图捕获的最大批大小 | None | — |
| `--cuda-graph-max-bs-prefill` | 预填充 CUDA 图捕获的最大批大小 | None | — |
| `--cuda-graph-max-seq-len-prefill` | 预填充 CUDA 图重放允许的最长序列；更长批次走 eager 预填充；并入 cuda_graph_config[prefill].max_seq_len | None | — |
| `--cuda-graph-bs-decode` | 解码 CUDA 图捕获的显式批大小列表 | None | — |
| `--cuda-graph-bs-prefill` | 预填充 CUDA 图捕获的显式批大小列表 | None | — |
| `--disable-cuda-graph-padding` | 需要 padding 时禁用 CUDA 图（无需 padding 时仍使用） | False | — |
| `--enable-profile-cuda-graph` | 启用 CUDA 图捕获的性能分析 | False | — |
| `--debug-cuda-graph` | 经 breakable 后端的 eager 模式 CUDA 图：图断点让每个算子 eager 执行但仍走捕获/重放路径；用于调试捕获/重放问题 | False | — |
| `--enable-cudagraph-gc` | CUDA 图捕获期间启用 GC；默认禁用（捕获时冻结 GC 以加速） | False | — |
| `--enable-layerwise-nvtx-marker` | 启用模型逐层 NVTX 性能分析标注；为每层添加 NVTX 标记，配合 Nsight Systems 做逐层性能分析 | False | — |
| `--enable-nccl-nvls` | 可用时为重预填充请求启用 NCCL NVLS | False | — |
| `--enable-symm-mem` | 启用 NCCL 对称内存加速集合通信 | False | — |
| `--disable-flashinfer-cutlass-moe-fp4-allgather` | 禁用 flashinfer cutlass MoE 的 all-gather 前量化 | False | — |
| `--enable-tokenizer-batch-encode` | 处理多文本输入时启用批量分词提升性能；勿与图像输入、预分词 input_ids 或 input_embeds 同用 | False | — |
| `--disable-tokenizer-batch-decode` | 解码多个补全时禁用批量反分词 | False | — |
| `--disable-outlines-disk-cache` | 禁用 outlines 磁盘缓存，避免文件系统或高并发相关的崩溃 | False | — |
| `--boundary-reduction` | 选择可选的 FFN 输出归约策略；不支持的 reduce-scatter 路径回退为 all-reduce 后 token 重分布；参见 FFN boundary reduction | auto | auto / ar / rs / rsv / rs+rsv |
| `--disable-custom-all-reduce` | 禁用自定义 all-reduce 内核，回退 NCCL | False | — |
| `--enable-mscclpp` | all-reduce 内核的小消息启用 mscclpp，失败回退 NCCL | False | — |
| `--enable-torch-symm-mem` | all-reduce 内核启用 torch symm mem，回退 NCCL；仅支持 CUDA SM90 以上：SM90 支持 world size 4、6、8；SM10 支持 6、8 | False | — |
| `--disable-overlap-schedule` | 禁用 overlap 调度器（CPU 调度与 GPU 模型 worker 重叠执行） | False | — |
| `--enable-mixed-chunk` | 使用分块预填充时允许批次内混合预填充与解码 | False | — |
| `--enable-dp-attention` | 已废弃：请改用 --attn-dp-size <dp-size>，而非 --dp-size <dp-size> --enable-dp-attention | False | — |
| `--enable-dp-lm-head` | 在注意力 TP 组内启用词表并行，避免跨 DP 组 all-gather，优化 DP attention 下的性能 | False | — |
| `--enable-two-batch-overlap` | 启用两个微批次重叠（TBO） | False | — |
| `--enable-single-batch-overlap` | 单个微批次内计算与通信重叠 | False | — |
| `--tbo-token-distribution-threshold` | 微批重叠中两批次 token 分布的阈值，决定走 two-batch-overlap 还是 two-chunk-overlap；设为 0 表示禁用 two-chunk-overlap | 0.48 | — |
| `--enable-torch-compile` | 用 torch.compile 优化模型；实验特性 | False | — |
| `--enable-torch-compile-debug-mode` | 启用 torch compile 调试模式 | False | — |
| `--torch-compile-max-bs` | 使用 torch compile 时的最大批大小 | 32 | — |
| `--disable-cuda-graph` | 已废弃：请改用 --cuda-graph-backend-decode=disabled 和/或 --cuda-graph-backend-prefill=disabled | False | — |
| `--disable-prefill-cuda-graph` | 禁用预填充阶段 CUDA 图；即 --cuda-graph-backend-prefill=disabled 的便捷写法 | False | — |
| `--disable-decode-cuda-graph` | 禁用解码阶段 CUDA 图；即 --cuda-graph-backend-decode=disabled 的便捷写法 | False | — |
| `--enable-p2p-check` | 启用 GPU P2P 访问检查；否则默认允许 p2p 访问 | False | — |
| `--triton-attention-reduce-in-fp32` | 将注意力中间结果转为 FP32，避免 FP16 相关崩溃；仅影响 Triton 注意力内核 | False | — |
| `--triton-attention-num-kv-splits` | flash decoding Triton 内核的 KV 切分数；长上下文场景更大的值更优 | 8 | — |
| `--triton-attention-split-tile-size` | flash decoding Triton 内核的 KV 切分块大小；用于确定性推理 | None | — |
| `--num-continuous-decode-steps` | 连续执行多个解码步以降低调度开销；可能提升吞吐但会增加首 token 延迟；默认 1（每次只执行一个解码步） | 1 | — |
| `--delete-ckpt-after-loading` | 加载模型后删除模型检查点 | False | — |
| `--enable-memory-saver` | 允许通过 release_memory_occupation 与 resume_memory_occupation 节省内存 | False | — |
| `--enable-weights-cpu-backup` | 在 release_weights_occupation 与 resume_weights_occupation 期间将模型权重保存到 CPU 内存 | False | — |
| `--enable-draft-weights-cpu-backup` | 在 release_weights_occupation 与 resume_weights_occupation 期间将 draft 模型权重保存到 CPU 内存 | False | — |
| `--allow-auto-truncate` | 允许自动截断超过最大输入长度的请求而非返回错误 | False | — |
| `--enable-custom-logit-processor` | 允许用户向服务器传入自定义 logit 处理器（默认出于安全考虑禁用） | False | — |
| `--enable-watermark` | 启用 Aaronson-Gumbel 文本水印 | False | — |
| `--watermark-config` | 水印设置：JSON 文件路径或内联 JSON 对象，可选 key、key_b、context_window、mixing_probability、max_probability、default_enabled、enforce_all 字段 | None | — |
| `--flashinfer-mla-disable-ragged` | 运行 flashinfer MLA 时不使用 ragged prefill 包装 | False | — |
| `--disable-shared-experts-fusion` | 禁用 DeepSeek V3/R1 的共享专家融合优化 | False | — |
| `--disable-chunked-prefix-cache` | 禁用 DeepSeek 的分块前缀缓存，短序列可节省开销 | False | — |
| `--image-processor-backend` | 图像处理器后端；auto 让 Transformers 选择最佳可用后端 | auto | auto / torchvision / pil |
| `--disable-fast-image-processor` | 已废弃：请改用 --image-processor-backend=pil | False | — |
| `--enable-return-hidden-states` | 启用随响应返回隐藏状态 | False | — |
| `--return-hidden-states-mode` | 服务器支持的最大隐藏状态返回模式；last 允许 return_hidden_states=False 或 last 的请求；full 还允许 return_hidden_states=True | None | last / full |
| `--enable-return-routed-experts` | 启用随响应返回每层的路由专家 | False | — |
| `--scheduler-recv-interval` | 调度器轮询请求的间隔；设为 >1 可降低轮询开销 | 1 | — |
| `--numa-node` | 设置子进程的 NUMA 节点；第 i 个元素对应第 i 个子进程 | None | — |
| `--enable-deterministic-inference` | 启用批次不变算子的确定性推理 | False | — |
| `--rl-on-policy-target` | SGLang 需要匹配以实现真 on-policy 的训练系统 | None | fsdp |
| `--enable-attn-tp-input-scattered` | 仅使用张量并行时允许注意力输入分散，降低 qkv latent 等算子的计算负载 | False | — |
| `--enable-prefill-cp` | 预填充阶段启用上下文并行；布局用 --cp-strategy 选择 | False | — |
| `--cp-strategy` | 预填充 CP 的分片策略：zigzag 每 rank 分配一个早块与一个晚块；interleave 按 CP 大小对 token 索引取模分配 | None | zigzag / interleave |
| `--enable-kv-cache-sharding` | 以页粒度对当前复制 KV缓存的组分片静态存储（GQA 模型 prefill CP 下的 attention-CP rank；MLA 模型的 attention-TP rank），成倍提升该组的唯一 KV 容量；仅限 PD 分离的预填充 worker | False | — |
| `--enable-fused-qk-norm-rope` | 启用融合的 QK 归一化与 RoPE 旋转位置编码 | False | — |
| `--enable-precise-embedding-interpolation` | 嵌入网格 resize 启用角对齐，确保插值嵌入值评估更准确（但更慢） | False | — |
| `--kv-canary` | KV缓存金丝雀模式：none 禁用（默认）；log 服务器持续运行时打印（生产安全）；raise 首次检测到不匹配即让服务器失败（CI 通道） | none | none / log / raise |
| `--kv-canary-real-data` | 金丝雀中检查真实 KV缓存：none 禁用（默认）；partial 检查每个真实 KV 槽的前 16 字节；all 检查完整真实 KV 槽 | none | — |
| `--kv-canary-sweep-interval` | 每 N 个前向步执行一次全池扫描 | 0 | — |
| `--pre-warm-nccl` | 启动期间预热 NCCL/RCCL 通信器，降低 P99 TTFT 冷启动延迟；默认 AMD/HIP（RCCL）启用、NVIDIA/CUDA（NCCL）禁用 | False | — |
| `--enable-dp-attention-local-control-broadcast` | DP attention 下控制消息发往每个 DP 组 leader 并在 attn_tp_group 内广播（而非全 tp_group），消除每次调度迭代的昂贵全 rank gloo 同步 | False | — |
| `--enforce-shared-experts-fusion` | 即使通常会被禁用（如 DeepEP 下）也强制共享专家融合；与 --disable-shared-experts-fusion 互斥 | False | — |
| `--enable-return-indexer-topk` | 启用随响应返回带索引器各层的 indexer topk 索引 | False | — |
| `--disable-attn-tp-gather` | 禁用调度器侧 attn_tp_gather（上游 SP 路径：将 num_tokens 填充到 attn_tp_size 并预分配 gathered buffer）；适用于在模型层自行管理 SP scatter/gather（如在注意力内做自己的 all_gather/reduce_scatter）且不消费上游 gathered_buffer 的模型；否则 CUDA 图运行器会把 num_tokens 填充到 attn_tp_size，小批次下可能导致内核自动调优器选错尺寸变体 | False | — |
| `--enable-fused-moe-sum-all-reduce` | 启用融合 MoE triton 与 sum all-reduce | False | — |
| `--gc-threshold` | 设置垃圾回收阈值（回收频率）；接受 1 到 3 个整数 | — | — |

### 3.28 动态批量分词器（Dynamic batch tokenizer）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--enable-dynamic-batch-tokenizer` | 启用异步动态批量分词器，多请求并发到达时提升性能 | False | — |
| `--dynamic-batch-tokenizer-batch-size` | [仅在设置 --enable-dynamic-batch-tokenizer 时使用] 动态批量分词器的最大批大小 | 32 | — |
| `--dynamic-batch-tokenizer-batch-timeout` | [仅在设置 --enable-dynamic-batch-tokenizer 时使用] 批处理分词请求的超时（秒） | 0.002 | — |

### 3.29 调试张量转储（Debug tensor dumps）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--debug-tensor-dump-output-folder` | 张量转储的输出目录 | None | — |
| `--debug-tensor-dump-layers` | 要转储的层 id；未指定则转储所有层 | None | — |
| `--debug-tensor-dump-input-file` | 张量转储的输入文件名 | None | — |

### 3.30 PD 分离（PD disaggregation）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--disaggregation-mode` | 仅 PD 分离使用：prefill 为纯预填充服务器，decode 为纯解码服务器；未指定则不做 PD 分离 | null | null / prefill / decode |
| `--disaggregation-transfer-backend` | PD 分离传输的后端；默认 mooncake | mooncake | mooncake / nixl / ascend / fake / mori / mooncake_tcp |
| `--disaggregation-bootstrap-port` | 预填充服务器上 bootstrap 服务器的端口；默认 8998 | 8998 | — |
| `--disaggregation-ib-device` | PD 分离传输的 InfiniBand 设备，接受单个（如 --disaggregation-ib-device mlx5_0）或逗号分隔多个（如 mlx5_0,mlx5_1）；默认 None，启用 mooncake 后端时自动检测 | None | — |
| `--disaggregation-decode-allocation-policy` | 何时为每请求预留解码 KV 槽；预填充与解码服务器需设置相同策略；要求与权衡见官方文档 | early | early / prefill_complete |
| `--disaggregation-decode-enable-offload-kvcache` | 解码服务器（PD 模式）启用异步 KV缓存卸载 | False | — |
| `--num-reserved-decode-tokens` | 新请求加入运行批次时为其预留内存的解码 token 数 | 512 | — |
| `--disaggregation-decode-polling-interval` | 解码服务器轮询请求的间隔；设为 >1 可降低轮询开销 | 1 | — |
| `--disaggregation-decode-enable-radix-cache` | 解码服务器（PD 模式）启用基数树缓存：缓存 KV 前缀避免冗余传输；与 --enable-hisparse、投机解码、--disaggregation-transfer-backend fake 不兼容 | False | — |
| `--optimistic-prefill-attempts` | 跳过 bootstrap 等待的乐观预填充前向次数 | 0 | — |

### 3.31 编码器预填充分离（Encode prefill disaggregation）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--encoder-only` | 对带编码器的 MLLM 启动纯编码器服务器 | False | — |
| `--language-only` | 对 VLM 只加载语言模型权重 | False | — |
| `--language-model-only` | 完全跳过多模态编码器：其权重不加载、塔不构建，为 KV缓存腾出 GPU 显存；拒绝多模态请求；与 --language-only 不同，这是独立模式而非编码器/解码器分离的一部分 | False | — |
| `--encoder-transfer-backend` | 编码器分离传输的后端；默认 zmq_to_scheduler | zmq_to_scheduler | zmq_to_scheduler / zmq_to_tokenizer / mooncake |
| `--encoder-urls` | 编码器服务器 URL 列表 | [] | — |
| `--encoder-bootstrap-port` | 运行在 language-only 分词器管理器进程中的 EncoderBootstrapServer 端口；编码器在此注册，language-only 接收方从此获取当前 URL 列表 | 8997 | — |
| `--encoder-register-urls` | 启动时注册本编码器的一个或多个 EncoderBootstrapServer URL，用于动态编码器发现；如 --encoder-register-urls http://prefill0:8997 http://prefill1:8997；与 --encoder-only 服务器配合使用 | [] | — |
| `--enable-adaptive-dispatch-to-encoder` | 启用自适应分发：language_only epd 模式下多图请求发给编码器，单图请求本地处理 | False | — |

### 3.32 自定义权重加载器（Custom weight loader）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--custom-weight-loader` | 用于更新模型的自定义数据加载器；应设置为有效的导入路径，如 my_package.weight_load_func | None | — |
| `--weight-loader-disable-mmap` | 用 safetensors 加载权重时禁用 mmap | False | — |
| `--weight-loader-prefetch-checkpoints` | 加载前将检查点文件预取进 OS 页缓存；每 rank 用后台线程预取一部分分片，把共享文件系统（NFS/Lustre）的总网络 IO 从 N*checkpoint 降到 1*checkpoint；推荐用于网络存储上的模型；启用时默认禁用多线程 safetensors 加载以避免与预取线程 IO 超订，可在 --model-loader-extra-config 设 enable_multithread_load=true 保留多线程加载（如本地 NVMe 上预取无意义时） | False | — |
| `--weight-loader-prefetch-num-threads` | 每 rank 检查点预取的线程数 | 4 | — |
| `--remote-instance-weight-loader-seed-instance-ip` | 从远端实例加载权重的种子实例 IP | None | — |
| `--remote-instance-weight-loader-seed-instance-service-port` | 从远端实例加载权重的种子实例服务端口 | None | — |
| `--remote-instance-weight-loader-send-weights-group-ports` | 从远端实例加载权重的通信组端口 | None | — |
| `--remote-instance-weight-loader-backend` | 从远端实例加载权重的后端：transfer_engine、nccl 或 modelexpress；默认 nccl | nccl | transfer_engine / nccl / modelexpress |
| `--remote-instance-weight-loader-start-seed-via-transfer-engine` | 远端实例权重加载器经 transfer engine 后端启动种子服务器 | False | — |
| `--weight-loader-drop-cache-after-load` | 每个 safetensors 分片加载后调用 posix_fadvise(DONTNEED) | False | — |
| `--engine-info-bootstrap-port` | 引擎信息 bootstrap 服务器的端口；默认 6789；同节点运行多实例时必须显式设置 | 6789 | — |
| `--modelexpress-config` | ModelExpress P2P 权重加载的 JSON 配置；键：url（可选的 gRPC host:port 覆盖）、transport（nixl 或 transfer_engine）；如 '{"url": "localhost:8001", "transport": "nixl"}' | None | — |

### 3.33 PD 多路复用（For PD-Multiplexing）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--enable-pdmux` | 启用 PD 多路复用：PD 运行于 greenctx 流 | False | — |
| `--pdmux-config-path` | PD 多路复用配置文件路径 | None | — |
| `--sm-group-num` | SM 分区组数 | 8 | — |

### 3.34 配置文件支持（Configuration file support）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--config` | 从配置文件读取 CLI 选项；须为含配置选项的 YAML 文件 | None | — |

### 3.35 多模态相关（For Multi-Modal）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--enable-broadcast-mm-inputs-process` | 调度器中启用广播 mm-inputs 进程 | False | — |
| `--mm-process-config` | 多模态预处理配置（JSON），含 image、video、audio 键 | {} | — |
| `--allowed-media-domains` | 将客户端提供的 HTTP(S) 媒体 URL 与重定向目标限制为这些精确主机名（空格分隔） | 无限制 | — |
| `--media-url-max-file-size-mb` | 单个远端媒体下载的最大流式大小（MiB）；设为 0 禁用限制 | 64 | — |
| `--mm-enable-dp-encoder` | 多模态编码器启用数据并行；dp size 会自动设为 tp size | False | — |
| `--limit-mm-data-per-request` | 每请求多模态输入的数量限制，如 '{"image": 1, "video": 1, "audio": 1}' | None | — |
| `--enable-mm-global-cache` | 编码器服务器启用 Mooncake 支持的全局多模态嵌入缓存，重复图像可复用缓存的 ViT 嵌入而非重新计算 | False | — |

### 3.36 检查点解密（For checkpoint decryption）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--decrypted-config-file` | 解密后的配置文件路径 | None | — |
| `--decrypted-draft-config-file` | 解密后的 draft 配置文件路径 | None | — |
| `--enable-prefix-mm-cache` | 启用前缀多模态缓存；目前仅支持 mm-only | False | — |

### 3.37 前向钩子（Forward hooks）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--forward-hooks` | JSON 格式的前向钩子规格列表；每个元素须包含 target_modules（与 model.named_modules() 名称匹配的 glob 模式列表）与 hook_factory（工厂的 Python 导入路径，如 my_package.hooks:make_hook）；可选 name 字段用于日志，可选 config 对象以字典形式传给工厂 | None | — |

### 3.38 MindStudio-probe（msProbe）转储（For MindStudio-probe(msProbe) dump）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--msprobe-dump-config` | msProbe 的 JSON 配置文件路径；指定即启用 msProbe 转储 | None | — |

