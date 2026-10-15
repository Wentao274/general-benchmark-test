# vLLM 与 SGLang bench（基准测试）参数详解大全

本文档系统整理 vLLM 与 SGLang 两大推理框架**基准测试（benchmark）命令行工具**的全部启动参数，涵盖功能说明、默认值与可选值，供性能测试与压测场景查阅。

> 同系列文档：《vLLM_SGLang_serve.md》整理了两大框架**服务部署（serve/launch_server）**的全部启动参数，可配合使用。

## 一、概述

### 1.1 工具矩阵

| 框架 | 命令 | 类型 | 说明 |
|---|---|---|---|
| vLLM | `vllm bench serve` | 在线压测（HTTP 客户端） | 对已启动的服务发起请求，测量吞吐、TTFT、ITL 等在线指标 |
| vLLM | `vllm bench throughput` | 离线测试（本地引擎） | 直接在本地引擎上离线跑数据集，测量总吞吐（token/s） |
| vLLM | `vllm bench latency` | 离线测试（本地引擎） | 单批 prefill/decode 延迟基准 |
| vLLM | `vllm bench forward` | 离线测试（本地引擎） | 仅测前向传播（pre fill / decode forward）性能 |
| SGLang | `python3 -m sglang.bench_serving` | 在线压测（HTTP 客户端） | 对已启动的服务发起请求，支持 sglang/vllm/lmdeploy 等多种后端 |
| SGLang | `python3 -m sglang.bench_one_batch` | 离线测试（本地引擎） | 不启动服务器，直接用底层 API 测单批 prefill/decode 延迟与吞吐 |

两框架的工具对应关系：

| 用途 | vLLM | SGLang |
|---|---|---|
| 在线服务压测 | `vllm bench serve` | `sglang.bench_serving` |
| 单批延迟测试 | `vllm bench latency` | `sglang.bench_one_batch` |
| 离线吞吐测试 | `vllm bench throughput` | `sglang.bench_serving`（在线模式测总吞吐） |

> **注意**：SGLang 新版本已将 bench 工具重构至 `sglang.benchmark` 包，`sglang.bench_serving` 旧调用方式已弃用，等价的新调用为 `python3 -m sglang.benchmark.serving`。

### 1.2 环境要求

- **vLLM**：bench 工具需安装额外依赖：`pip install vllm[bench]`
- **SGLang**：需要 Python 3.11+，依赖 aiohttp、numpy、requests、tqdm、transformers、datasets、pillow、pybase64 等（`pip install "sglang[all]"` 一般已包含）
- 两个工具均通过 `--help` 查看完整参数：`vllm bench serve --help`、`python3 -m sglang.bench_serving --help`

### 1.3 核心指标说明

| 指标 | 含义 |
|---|---|
| Throughput（吞吐量） | 每秒完成的输出 token 数（output token/s）或每秒完成请求数（request/s） |
| TTFT（Time To First Token） | 首 token 延迟 |
| ITL（Inter-Token Latency） | token 间延迟，即两次流式输出之间的间隔 |
| TPOT（Time Per Output Token） | 每输出 token 平均时间（不含首 token） |
| E2EL（End-to-End Latency） | 单请求端到端总延迟 |
| Goodput（有效吞吐） | 满足延迟 SLO 约束的有效吞吐量（SLO 相关仅 `vllm bench serve` 支持） |


## 二、vLLM bench 参数详解

vLLM 提供 `vllm bench {forward, latency, serve, throughput}` 四个基准测试子命令，分别用于前向传播、单批延迟、在线服务压测与离线吞吐量测试。使用前需通过 `pip install vllm[bench]` 安装额外依赖。

### 2.1 vllm bench serve（在线服务基准测试）

`vllm bench serve` 是独立的 HTTP 客户端压测工具，用于对运行中的 OpenAI 兼容推理服务发起请求，测量 TTFT、TPOT、ITL、E2EL、吞吐量与 goodput 等服务级指标。其参数自成一体，不依赖引擎参数。

```bash
vllm bench serve \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --backend vllm \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3.1_unfiltered_condensed_new.json \
    --num-prompts 1000 \
    --request-rate 10 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result
```

#### 2.1.1 通用参数

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--trust-remote-code` | 信任来自 HuggingFace 的远程代码 | `False` | — |
| `--seed` | 随机种子 | `0` | — |
| `--num-prompts` | 处理的请求（prompt）数量 | `1000` | — |
| `--dataset-name` | 基准测试使用的数据集名称 | `random` | sharegpt / burstgpt / random / random-mm / random-rerank / hf / custom / custom_audio / custom_image / prefix_repetition / spec_bench / speed_bench / timed_trace |
| `--no-stream` | 不以流式模式加载数据集 | `False` | — |
| `--dataset-path` | ShareGPT 数据集路径；使用 HF 数据集时为 HF 数据集 ID | — | — |
| `--no-oversample` | 数据集样本数少于 num-prompts 时不做过采样 | `False` | — |
| `--skip-chat-template` | 对支持的数据集跳过对 prompt 应用聊天模板 | `False` | — |
| `--enable-multimodal-chat` | 对支持的数据集启用多模态聊天变换 | `False` | — |
| `--disable-shuffle` | 禁用数据集样本打乱，保证确定性顺序 | `False` | — |
| `--label` | 基准测试结果的标签（前缀）；未指定时使用 `--backend` 的值作为标签 | — | — |
| `--backend` | 基准测试使用的后端或端点类型 | `openai` | vllm / openai / openai-chat / openai-responses / openai-audio / openai-embeddings / openai-embeddings-chat / openai-embeddings-clip / openai-embeddings-vlm2vec / infinity-embeddings / infinity-embeddings-clip / vllm-pooling / vllm-rerank |
| `--base-url` | 不使用 host/port 时，服务器或 API 的 base URL | — | — |
| `--host` | 服务器主机地址 | `127.0.0.1` | — |
| `--port` | 服务器端口 | `8000` | — |
| `--endpoint` | API 端点 | `/v1/completions` | — |
| `--header` | 以键值对（如 `--header x-additional-info=0.3.3`）为每个请求传递的 header；会覆盖各后端常量及环境变量设置的值，但会被其他参数（如 request id）覆盖 | — | — |
| `--max-concurrency` | 最大并发请求数，用于模拟上层组件限制并发的场景；`--request-rate` 控制请求发起速率，本参数控制同时实际执行的请求数，二者组合时若服务器处理不过来，实际请求速率可能低于指定值 | — | — |
| `--model` | 模型名称；未指定时从服务器 `/v1/models` 端点获取第一个模型 | — | — |
| `--input-len` | 数据集的通用输入长度，映射到 `--random-input-len`；未指定时使用数据集默认值 | — | — |
| `--output-len` | 数据集的通用输出长度，映射到数据集特定的输出长度参数（如 `--random-output-len`、`--hf-output-len`）；未指定时使用数据集默认值 | — | — |
| `--tokenizer` | 不使用默认分词器时，分词器的名称或路径 | — | — |
| `--tokenizer-mode` | 分词器模式：auto（Mistral 模型优先使用 mistral_common，否则使用 HF）、hf（可用时使用 fast 分词器）、mistral（强制使用 mistral_common）、deepseek_v32（强制使用 deepseek_v32）；其他自定义值可通过插件支持 | `auto` | auto / hf / mistral / deepseek_v32 |
| `--use-beam-search` | 使用束搜索 | `False` | — |
| `--logprobs` | 每个 token 计算并返回的 logprobs 数量；未指定时：若禁用束搜索则不计算 logprobs（每个 token 返回一个 dummy logprob），若启用束搜索则每个 token 计算 1 个 logprob | — | — |
| `--request-rate` | 每秒请求数；为 inf 时所有请求在时刻 0 同时发出，否则使用 Poisson 过程或 gamma 分布合成请求到达时间 | `inf` | — |
| `--burstiness` | 请求生成的突发度因子，仅在 request-rate 非 inf 时生效；默认 1 为 Poisson 过程，否则请求间隔服从 gamma 分布；值越接近 0（0 到 1 之间）请求越突发，值大于 1 请求到达越均匀 | `1.0` | — |
| `--probe-request-rate` | 为正时以该速率（req/s）随主负载发送单 token 纯文本探测请求（绕过 `--max-concurrency`），并单独报告其延迟；用于测量主负载对无关请求的拖慢程度 | `0.0` | — |
| `--disable-tqdm` | 禁用 tqdm 进度条 | `False` | — |
| `--num-warmups` | 预热请求数 | `0` | — |
| `--profile` | 使用 vLLM 性能分析；服务器端必须提供 `--profiler-config` | `False` | — |
| `--save-result` | 将基准测试结果保存到 JSON 文件 | `False` | — |
| `--save-detailed` | 保存结果时是否包含每请求的详细信息（如 response、error、ttfts、tpots 等） | `False` | — |
| `--append-result` | 将基准测试结果追加到已存在的 JSON 文件 | `False` | — |
| `--metadata` | 以键值对（如 `--metadata version=0.3.3 tp=1`）保存到结果 JSON 文件中的本次运行元数据，便于记录 | — | — |
| `--result-dir` | 保存基准测试 JSON 结果的目录；未指定时保存到当前目录 | — | — |
| `--result-filename` | 保存基准测试 JSON 结果的文件名；未指定时按 `{label}-{request_rate}qps-{base_model_id}-{current_dt}.json` 格式命名 | — | — |
| `--ignore-eos` | 发送基准测试请求时设置 ignore_eos 标志；警告：deepspeed_mii 和 tgi 不支持 ignore_eos | `False` | — |
| `--self-timed` / `--no-self-timed` | 使用 trace 自带的时间信息而非配置，适用于按时间戳忠实回放 trace；默认 False，但 `--dataset-name=timed_trace` 时默认 True，可用 `--no-self-timed` 强制关闭；关闭时使用用户定义的生成速率并忽略 trace 内计时信息 | `False`（timed_trace 数据集下默认 `True`） | — |
| `--percentile-metrics` | 逗号分隔的需要报告百分位数的指标；允许的指标名为 ttft / tpot / itl / e2el / client_queue_time / e2el_including_client_queue | 生成模型为 `ttft,tpot,itl`，pooling 模型为 `e2el` | — |
| `--metric-percentiles` | 所选指标的百分位数，逗号分隔，如 "25,50,75" 表示报告第 25/50/75 百分位；需配合 `--percentile-metrics` 选择指标 | `99` | — |
| `--goodput` | 以 "KEY:VALUE" 对指定 goodput 的服务级目标（SLO），key 为指标名，value 为毫秒，可空格分隔提供多对；允许的请求级指标为 ttft / tpot / e2el；goodput 定义参见 DistServe 论文 | — | — |
| `--request-id-prefix` | request id 的前缀 | `bench-ce9b7f98-` | — |
| `--served-model-name` | API 中使用的模型名；未指定时与 `--model` 参数相同 | — | — |
| `--lora-modules` | 启动服务器时传入的 LoRA 模块名称子集；默认每个请求随机选择一个 LoRA 模块，可用 `--lora-assignment` 控制选择策略 | — | — |
| `--lora-assignment` | 为请求分配 LoRA 模块的策略：random 为每请求随机选择，round-robin 为确定性轮询 | `random` | random / round-robin |
| `--ramp-up-strategy` | 爬坡策略，在基准测试持续时间内将请求速率从起始 RPS 爬升到结束 RPS（由 `--ramp-up-start-rps`、`--ramp-up-end-rps` 指定） | — | linear / exponential |
| `--ramp-up-start-rps` | 爬坡的起始请求速率（RPS）；使用 `--ramp-up-strategy` 时必须指定 | — | — |
| `--ramp-up-end-rps` | 爬坡的结束请求速率（RPS）；使用 `--ramp-up-strategy` 时必须指定 | — | — |
| `--ready-check-timeout-sec` | 等待端点就绪的最长时间（秒）；默认跳过就绪检查 | `0` | — |
| `--chat-template-kwargs` | JSON 字符串，数据集在客户端渲染 prompt（如 custom / speed_bench）时转发给分词器 apply_chat_template 的 kwargs，如 '{"thinking": true}' 启用推理模式 | — | — |
| `--extra-body` | JSON 字符串，随每个请求发送的额外 body 参数，如 '{"chat_template_kwargs":{"enable_thinking":false}}' | — | — |
| `--skip-tokenizer-init` | 跳过分词器与去分词器的初始化 | `False` | — |
| `--insecure` | 禁用 SSL 证书验证，用于连接使用自签名证书的服务器 | `False` | — |
| `--plot-timeline` | 生成展示请求执行过程的 HTML 时间线图，保存在结果 JSON 文件旁 | `False` | — |
| `--timeline-itl-thresholds` | 时间线图着色的 ITL 阈值（毫秒），两个逗号分隔的值将 ITL 分为三组：低于第一阈值（绿色）、介于两阈值之间（橙色）、高于第二阈值（红色） | `25,50` | — |
| `--plot-dataset-stats` | 生成展示数据集统计（prompt token、输出 token 及合并 token 分布）的 matplotlib 图 | `False` | — |

#### 2.1.2 custom 数据集选项

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--custom-output-len` | 每个请求的输出 token 数，仅用于 custom 数据集；除非设为 -1，否则覆盖数据集中潜在的输出长度 | `256` | — |
| `--custom-ensure-client-side-data` | 确保 custom 数据集的媒体以客户端数据而非引用形式发送；对 custom_image 数据集，会在基准测试客户端加载本地及 HTTP(S) 图片并编码为 base64 data URL，已有的 data:image URL 保持不变 | `False` | — |

#### 2.1.3 spec bench 数据集选项

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--spec-bench-output-len` | 每个请求的输出 token 数，仅用于 spec bench 数据集 | `256` | — |
| `--spec-bench-category` | spec bench 数据集的类别；为 None 时使用所有类别 | `None` | — |

#### 2.1.4 sharegpt 数据集选项

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--sharegpt-output-len` | 每个请求的输出长度，覆盖 ShareGPT 数据集的输出长度 | — | — |

#### 2.1.5 timed-trace 数据集选项

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--timed-trace-chunk-hash-size` | 每个 hash token（若存在）代表多少个 token hash；例如 Moonshot trace 中为 512，Qwen/Alibaba 为 16 | `16` | — |
| `--timed-trace-sec-multiplier` | 时间戳转秒时使用的乘数，将时间戳乘以该值；例如时间戳为毫秒时传 0.001，已为秒时默认 1 即可 | `1` | — |
| `--timed-trace-label-timestamp` | 用于索引 trace 中时间戳的 JSON 标签 | `timestamp` | — |
| `--timed-trace-label-input-length` | 用于索引 trace 中输入长度字段的 JSON 标签 | `input_length` | — |
| `--timed-trace-label-output-length` | 用于索引 trace 中输出长度字段的 JSON 标签 | `output_length` | — |
| `--timed-trace-label-hash-ids` | 用于索引输入 prompt 的 hash id 的 JSON 标签 | `hash_ids` | — |

#### 2.1.6 blazedit 数据集选项

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--blazedit-min-distance` | blazedit 数据集的最小距离（范围 0 至 1.0） | `0.0` | — |
| `--blazedit-max-distance` | blazedit 数据集的最大距离（范围 0 至 1.0） | `1.0` | — |

#### 2.1.7 asr 数据集选项

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--asr-max-audio-len-sec` | ASR 数据集的最大音频时长（秒） | `inf` | — |
| `--asr-min-audio-len-sec` | ASR 数据集的最小音频时长（秒） | `0.0` | — |

#### 2.1.8 random 数据集选项

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--random-input-len` | 每个请求的输入 token 数，仅用于随机采样 | `1024` | — |
| `--random-output-len` | 每个请求的输出 token 数，仅用于随机采样 | `128` | — |
| `--random-range-ratio` | 采样输入/输出长度的范围比例，仅用于随机采样；单个浮点数同时作用于 ISL 与 OSL，JSON 字典（如 '{"input": 0.3, "output": 0.5}'）可分别设置；取值须在 [0, 1) 内 | `0.0` | — |
| `--random-prefix-len` | 请求中随机上下文之前的固定前缀 token 数；总输入长度为 random-prefix-len 与从 [input_len*(1-range_ratio), input_len*(1+range_ratio)] 采样的随机上下文长度之和 | `0` | — |
| `--random-batch-size` | 随机采样的批大小，仅用于 embeddings 基准测试 | `1` | — |
| `--no-reranker` | 模型是否原生支持重排序，仅用于 reranker 基准测试 | `False` | — |

#### 2.1.9 random 多模态数据集选项（扩展自 random 数据集）

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--random-mm-base-items-per-request` | random-mm 每个请求的多模态 item 基数；实际每请求的数量基于该基数按 `--random-mm-num-mm-items-range-ratio` 采样 | `1` | — |
| `--random-mm-num-mm-items-range-ratio` | 每请求 item 数的采样范围比例 r（[0, 1]）：从闭整数区间 [floor(n*(1-r)), ceil(n*(1+r))]（n 为每请求 item 基数）均匀采样；r=0 保持固定，r=1 允许 0 个 item；最大值被 `--random-mm-limit-mm-per-prompt` 各模态上限之和截断，若计算出的最小值超过最大值则报错 | `0.0` | — |
| `--random-mm-limit-mm-per-prompt` | 每请求附加 item 的各模态硬上限，如 '{"image": 3, "video": 0}'；采样的每请求 item 数被这些上限之和截断；某模态达到上限后其桶被排除且概率重新归一化；注：目前仅支持图片采样 | `{'image': 255, 'video': 1}` | — |
| `--random-mm-bucket-config` | 桶配置：将多模态 item 采样配置映射到概率的字典，桶键为 (height, width, num_frames) 元组，值为采样该 item 的概率；当前支持图片和视频两种模态，概率之和不为 1 时自动归一化；注：目前仅支持图片采样 | `{(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}` | — |

#### 2.1.10 hf 数据集选项

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--hf-subset` | HF 数据集的子集 | — | — |
| `--hf-split` | HF 数据集的切分 | — | — |
| `--hf-name` | HuggingFace 上数据集的名称（如 'lmarena-ai/VisionArena-Chat'）；当 dataset-path 为本地路径时需指定 | — | — |
| `--hf-output-len` | 每个请求的输出长度，覆盖采样 HF 数据集的输出长度 | — | — |

#### 2.1.11 BFCL 数据集选项

BFCL（Berkeley Function Calling Leaderboard）数据集（https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard）用于工具调用流量测试：加载器按类别下载 JSON-lines 文件并轮询交织行以均衡采样，将 BFCL 函数 schema 转换为 OpenAI tool 格式，配合 openai-chat 后端产生贴近生产的工具调用流量。

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--bfcl-categories` | 逗号分隔的 BFCL v3 类别名列表（不带 'BFCL_v3_' 前缀和 '.json' 后缀），如 'simple,live_simple,multiple' | `simple,live_simple,multiple` | — |

#### 2.1.12 prefix repetition 数据集选项

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--prefix-repetition-prefix-len` | 每个请求的前缀 token 数，仅用于 prefix repetition 数据集 | `256` | — |
| `--prefix-repetition-suffix-len` | 每个请求的后缀 token 数，仅用于 prefix repetition 数据集；总输入长度为 prefix_len + suffix_len | `256` | — |
| `--prefix-repetition-num-prefixes` | 生成的前缀数量，仅用于 prefix repetition 数据集；每个前缀的 prompt 数为 num_requests // num_prefixes | `10` | — |
| `--prefix-repetition-output-len` | 每个请求的输出 token 数，仅用于 prefix repetition 数据集 | `128` | — |

#### 2.1.13 speed bench 数据集选项

SPEED-Bench 数据集（https://huggingface.co/datasets/nvidia/SPEED-Bench），需先用官方脚本（NVIDIA-NeMo/Skills 仓库中的 prepare.py）下载。

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--speed-bench-dataset-subset` | SPEED-Bench 数据集的子集 | `qualitative` | throughput_16k / qualitative / throughput_32k / throughput_2k / throughput_8k / throughput_1k |
| `--speed-bench-output-len` | 每个请求的输出 token 数，仅用于 speed bench 数据集 | `4096` | — |
| `--speed-bench-category` | speed bench 数据集的类别；为 None 时使用所有类别 | `None` | — |

#### 2.1.14 采样参数

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--top-p` | Top-p 采样参数，仅对 openai 兼容后端生效 | — | — |
| `--top-k` | Top-k 采样参数，仅对 openai 兼容后端生效 | — | — |
| `--min-p` | Min-p 采样参数，仅对 openai 兼容后端生效 | — | — |
| `--temperature` | 温度采样参数，仅对 openai 兼容后端生效 | — | — |
| `--frequency-penalty` | 频率惩罚采样参数，仅对 openai 兼容后端生效 | — | — |
| `--presence-penalty` | 存在惩罚采样参数，仅对 openai 兼容后端生效 | — | — |
| `--repetition-penalty` | 重复惩罚采样参数，仅对 openai 兼容后端生效 | — | — |

### 2.2 vllm bench throughput（离线吞吐量基准测试）

`vllm bench throughput` 通过离线引擎直接处理整个数据集，测量引擎的端到端生成吞吐量（tokens/s）。

```bash
vllm bench throughput \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3.1_unfiltered_condensed_new.json \
    --num-prompts 1000
```

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--backend` | 基准测试使用的后端类型 | `vllm` | vllm / hf / mii / vllm-chat |
| `--dataset-name` | 基准测试使用的数据集名称 | `sharegpt` | sharegpt / random / burstgpt / hf / prefix_repetition / random-mm / random-rerank / custom_audio |
| `--dataset-path` | 数据集路径 | — | — |
| `--no-oversample` | 数据集样本数少于 num-prompts 时不做过采样 | `False` | — |
| `--enable-multimodal-chat` | 对支持的数据集启用多模态聊天变换 | `False` | — |
| `--custom-output-len` | custom 数据集每个请求的输出 token 数 | — | — |
| `--input-len` | 每个请求的输入 prompt 长度 | — | — |
| `--output-len` | 每个请求的输出长度，覆盖数据集的输出长度 | — | — |
| `--n` | 每个 prompt 生成的序列数 | `1` | — |
| `--num-prompts` | 处理的 prompt 数量 | `1000` | — |
| `--num-warmups` | 计时基准测试前处理的预热 prompt 数 | `0` | — |
| `--hf-max-batch-size` | HF 后端的最大批大小 | — | — |
| `--hf-enable-torch-compile` | 为 HF 后端启用 Torch compile | `False` | — |
| `--output-json` | 以 JSON 格式保存吞吐量结果的路径 | — | — |
| `--async-engine` | 使用 vLLM 异步引擎而非 LLM 类 | `False` | — |
| `--prequeue-requests` | 对 vLLM 后端，在允许调度器处理前将所有请求入队；可消除请求渲染与引擎调度的重叠以提高基准可复现性，但可能降低测得的吞吐量；请求渲染通常远快于调度和处理，该标志的预期用例是耗时图片渲染的多模态基准测试 | `False` | — |
| `--disable-detokenize` | 不对响应做去分词（即测量中不含去分词时间） | `False` | — |
| `--lora-path` | 要使用的 LoRA 适配器路径，可为绝对路径、相对路径或 Hugging Face 模型标识 | — | — |
| `--lora-assignment` | 为请求分配 LoRA 适配器的策略：random（默认）为每请求随机选择，round-robin 为确定性轮询 | `random` | random / round-robin |
| `--prefix-len` | 请求中随机上下文之前的固定前缀 token 数 | `0` | — |
| `--hf-subset` | HF 数据集的子集 | — | — |
| `--hf-split` | HF 数据集的切分 | — | — |
| `--hf-name` | HuggingFace 上数据集的名称（如 'lmms-lab/LLaVA-OneVision-Data'）；当 `--dataset-path` 为本地文件系统路径时指定，以便基准测试识别正确的数据集类 | — | — |
| `--profile` | 使用 vLLM 性能分析；服务器端必须提供 `--profiler-config` | `False` | — |
| `--prefix-repetition-prefix-len` | 每个请求的前缀 token 数，仅用于 prefix repetition 数据集 | — | — |
| `--prefix-repetition-suffix-len` | 每个请求的后缀 token 数，仅用于 prefix repetition 数据集；总输入长度为 prefix_len + suffix_len | — | — |
| `--prefix-repetition-num-prefixes` | 生成的前缀数量，仅用于 prefix repetition 数据集；每个前缀的 prompt 数为 num_requests // num_prefixes | — | — |
| `--prefix-repetition-output-len` | 每个请求的输出 token 数，仅用于 prefix repetition 数据集 | — | — |
| `--random-input-len` | 每个请求的输入 token 数，仅用于随机采样 | `1024` | — |
| `--random-output-len` | 每个请求的输出 token 数，仅用于随机采样 | `128` | — |
| `--random-range-ratio` | 采样输入/输出长度的范围比例，仅用于随机采样；单个浮点数同时作用于 ISL 与 OSL，JSON 字典（如 '{"input": 0.3, "output": 0.5}'）可分别设置；取值须在 [0, 1) 内 | `0.0` | — |
| `--random-prefix-len` | 请求中随机上下文之前的固定前缀 token 数；总输入长度为 random-prefix-len 与从 [input_len*(1-range_ratio), input_len*(1+range_ratio)] 采样的随机上下文长度之和 | `0` | — |
| `--random-batch-size` | 随机采样的批大小，仅用于 embeddings 基准测试 | `1` | — |
| `--no-reranker` | 模型是否原生支持重排序，仅用于 reranker 基准测试 | `False` | — |
| `--random-mm-base-items-per-request` | random-mm 每个请求的多模态 item 基数；实际每请求的数量基于该基数按 `--random-mm-num-mm-items-range-ratio` 采样 | `1` | — |
| `--random-mm-num-mm-items-range-ratio` | 每请求 item 数的采样范围比例 r（[0, 1]）：从闭整数区间 [floor(n*(1-r)), ceil(n*(1+r))]（n 为每请求 item 基数）均匀采样；r=0 保持固定，r=1 允许 0 个 item；最大值被 `--random-mm-limit-mm-per-prompt` 各模态上限之和截断，若计算出的最小值超过最大值则报错 | `0.0` | — |
| `--random-mm-limit-mm-per-prompt` | 每请求附加 item 的各模态硬上限，如 '{"image": 3, "video": 0}'；采样的每请求 item 数被这些上限之和截断；某模态达到上限后其桶被排除且概率重新归一化；注：目前仅支持图片采样 | `{'image': 255, 'video': 1}` | — |
| `--random-mm-bucket-config` | 桶配置：将多模态 item 采样配置映射到概率的字典，桶键为 (height, width, num_frames) 元组，值为采样该 item 的概率；当前支持图片和视频两种模态，概率之和不为 1 时自动归一化；注：目前仅支持图片采样 | `{(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}` | — |
| `--asr-min-audio-len-sec` | ASR 数据集过滤的最小音频时长（秒） | `0.0` | — |
| `--asr-max-audio-len-sec` | ASR 数据集过滤的最大音频时长（秒） | `inf` | — |
| `--disable-log-stats` | 禁用统计日志 | `False` | — |
| `--aggregate-engine-logging` | 使用数据并行时记录聚合而非各引擎的统计信息 | `False` | — |
| `--fail-on-environ-validation` / `--no-fail-on-environ-validation` | 若设置，环境校验失败时引擎将报错 | `False` | — |
| `--shutdown-timeout` | 关闭超时（秒）；0 表示中止，大于 0 表示等待 | `0` | — |
| `--gdn-prefill-backend` | 选择 GDN prefill 后端 | — | flashinfer / triton / cutedsl / aiter_flydsl |
| `--kda-prefill-backend` | 选择 KDA prefill 后端；flashkda 仅限 CUDA，fused 仅限 ROCm，auto 自动选择受支持的后端 | — | auto / triton / flashkda / flashinfer / fused |
| `--kda-decode-backend` | 选择 KDA decode 后端 | — | auto / native / flashinfer / triton |
| `--enable-log-requests` / `--no-enable-log-requests` | 启用请求信息日志，依赖日志级别：INFO 记录请求 ID、参数和 LoRA 请求，DEBUG 记录 prompt 输入（如文本、token ID）；可通过 VLLM_LOGGING_LEVEL 设置最低日志级别 | `False` | — |

> 此外，该命令还支持与 `vllm serve` 完全相同的引擎参数（ModelConfig、LoadConfig、ParallelConfig、CacheConfig、SchedulerConfig 等），详见《vLLM_SGLang_serve.md》第二章。

### 2.3 vllm bench latency（单批延迟基准测试）

`vllm bench latency` 以固定的批大小和输入/输出长度在离线引擎上运行多次迭代，测量单批生成的每 token 延迟（含 prefill/decode 各阶段耗时）。

```bash
vllm bench latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input-len 32 --output-len 128 \
    --batch-size 8 --num-iters 30
```

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--input-len` | 输入 prompt 长度 | `32` | — |
| `--output-len` | 输出长度 | `128` | — |
| `--batch-size` | 批大小（每次迭代处理的请求数） | `8` | — |
| `--n` | 每个 prompt 生成的序列数 | `1` | — |
| `--use-beam-search` | 使用束搜索 | `False` | — |
| `--num-iters-warmup` | 预热迭代次数 | `10` | — |
| `--num-iters` | 正式测量的迭代次数 | `30` | — |
| `--profile` | 对单批生成过程进行性能分析 | `False` | — |
| `--output-json` | 以 JSON 格式保存延迟结果的路径 | — | — |
| `--disable-detokenize` | 不对响应做去分词（即延迟测量中不含去分词时间） | `False` | — |
| `--disable-log-stats` | 禁用统计日志 | `False` | — |
| `--aggregate-engine-logging` | 使用数据并行时记录聚合而非各引擎的统计信息 | `False` | — |
| `--fail-on-environ-validation` / `--no-fail-on-environ-validation` | 若设置，环境校验失败时引擎将报错 | `False` | — |
| `--shutdown-timeout` | 关闭超时（秒）；0 表示中止，大于 0 表示等待 | `0` | — |
| `--gdn-prefill-backend` | 选择 GDN prefill 后端 | — | flashinfer / triton / cutedsl / aiter_flydsl |
| `--kda-prefill-backend` | 选择 KDA prefill 后端；flashkda 仅限 CUDA，fused 仅限 ROCm，auto 自动选择受支持的后端 | — | auto / triton / flashkda / flashinfer / fused |
| `--kda-decode-backend` | 选择 KDA decode 后端 | — | auto / native / flashinfer / triton |

> 此外，该命令还支持与 `vllm serve` 完全相同的引擎参数（ModelConfig、LoadConfig、ParallelConfig、CacheConfig、SchedulerConfig 等），详见《vLLM_SGLang_serve.md》第二章。

### 2.4 vllm bench forward（前向传播基准测试）

`vllm bench forward` 以合成请求形状直接测量模型前向传播性能（绕过采样与调度逻辑），适用于 kernel 与引擎配置调优。该命令默认以 dummy 权重加载模型（引擎参数 `--load-format` 默认值为 `dummy`），无需真实权重即可测试。

```bash
vllm bench forward \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --case prefill=8192 \
    --case decode=32*1@8192 \
    --iters 20 --samples 5
```

| 参数 | 说明 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| `--case` | 以 NAME=REQS 定义测试用例，REQS 为 [N*]Q[@K][c][:D] 的逗号列表（Q 为查询长度、K 为含查询在内的 KV 长度（默认 Q）、D 为草稿 token 数、N 为请求数）；prefill 的前缀默认为前缀缓存命中，c 标记分块 prefill；如 decode=32*1@8192、verify=16*4@4096:3、prefill=8192、hit=2048@10240、mix=8192,32*1@8192；可重复指定 | `[]` | — |
| `--shapes` | 包含测试用例的 JSON 或 YAML 文件 | — | — |
| `--kv-len-jitter` | 在其值的 ±该比例内均匀抽取每个请求的前缀长度（KV 长度减查询长度），模拟真实 batch 的长度差异，上限为模型最大长度；仅由 `--seed` 播种，每个 DP rank 得到相同长度；0 保持精确长度 | `0.5` | — |
| `--kv-init` | 每个用例各 block 的初始 KV cache 内容 | `zeros` | zeros |
| `--routing` | MoE 路由方式；balanced 为每个 EP rank 负载相等的固定路由，auto 在使用 dummy 权重时为 balanced、否则为 real | `auto` | auto / real / balanced |
| `--iters` | 每个样本的重放次数 | `20` | — |
| `--samples` | 采样次数 | `5` | — |
| `--output-json` | 将结果写入该文件 | — | — |
| `--disable-log-stats` | 禁用统计日志 | `False` | — |
| `--aggregate-engine-logging` | 使用数据并行时记录聚合而非各引擎的统计信息 | `False` | — |
| `--fail-on-environ-validation` / `--no-fail-on-environ-validation` | 若设置，环境校验失败时引擎将报错 | `False` | — |
| `--shutdown-timeout` | 关闭超时（秒）；0 表示中止，大于 0 表示等待 | `0` | — |
| `--gdn-prefill-backend` | 选择 GDN prefill 后端 | — | flashinfer / triton / cutedsl / aiter_flydsl |
| `--kda-prefill-backend` | 选择 KDA prefill 后端；flashkda 仅限 CUDA，fused 仅限 ROCm，auto 自动选择受支持的后端 | — | auto / triton / flashkda / flashinfer / fused |
| `--kda-decode-backend` | 选择 KDA decode 后端 | — | auto / native / flashinfer / triton |

> 此外，该命令还支持与 `vllm serve` 完全相同的引擎参数（ModelConfig、LoadConfig、ParallelConfig、CacheConfig、SchedulerConfig 等），详见《vLLM_SGLang_serve.md》第二章。

## 三、SGLang bench 参数详解

SGLang 提供两个核心基准测试工具：`sglang.bench_serving`（在线服务压测，作为 HTTP 客户端向已启动的服务发送请求，测量吞吐、TTFT、ITL、TPOT、E2E 延迟等）与 `sglang.bench_one_batch`（无服务器的单批静态延迟测试，直接使用底层 API，测量 prefill/decode 延迟与吞吐）。此外还有 `sglang.bench_one_batch_server`（bench_one_batch 的服务器模式变体），本章不展开。

### 3.1 sglang.bench_serving（在线服务基准测试）

`sglang.bench_serving`（旧路径 `python3 -m sglang.bench_serving`，新路径 `python3 -m sglang.benchmark.serving`）是面向在线推理服务的压测工具：它以 HTTP 客户端身份向一个已在运行的推理服务按指定速率发送请求，并统计吞吐量、TTFT（首 token 延迟）、ITL（token 间延迟）、TPOT（每输出 token 平均延迟）、E2E 延迟等指标。支持的后端包括：sglang / sglang-native / sglang-oai / sglang-oai-chat / sglang-embedding / vllm / vllm-chat / vllm-embedding / lmdeploy / lmdeploy-chat / trt / gserver / truss。可用 `python3 -m sglang.bench_serving --help` 查看全部参数。

典型用法示例：

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_clean_lang_split.json \
    --num-prompts 1000 \
    --request-rate 10 \
    --output-file bench_result.jsonl
```

#### 3.1.1 连接与后端

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --backend | 指定后端（取决于 LLM 推理引擎），决定请求协议与 API 路径 | sglang | sglang / sglang-native / sglang-oai / sglang-oai-chat / sglang-embedding / vllm / vllm-chat / vllm-embedding / lmdeploy / lmdeploy-chat / trt / gserver / truss |
| --base-url | 服务端或 API 基础 URL；设置后优先于 host/port | — | — |
| --host | 服务器主机地址 | 0.0.0.0 | — |
| --port | 服务器端口；未设置时按后端取默认值（sglang 系列为 30000、vllm/trt 为 8000、lmdeploy 为 23333、gserver 为 9988、truss 为 8080） | — | — |
| --ready-check-timeout-sec | 等待服务器就绪的最大秒数，设为 0 跳过就绪检查 | 60 | — |
| --model | 模型名称或路径；未设置时通过 /v1/models 接口获取模型配置 | — | — |
| --served-model-name | 服务端实际提供的模型名；未设置时默认取 --model 的值 | — | — |
| --tokenizer | tokenizer 名称或路径；未设置时使用模型配置 | — | — |
| --header | 自定义 HTTP 请求头，Key=Value 格式，可指定多个，如 --header MyHeader=MY_VALUE MyAnotherHeader=myanothervalue | — | — |
| --rust-client | 通过 Rust 客户端运行压测，每个响应流在独立线程上解析（而非共享单个 asyncio 线程）；指标与结果字段相同，但支持的后端和数据集较少（详见 rust/sglang-bench/README.md） | False | — |
| --rust-worker-threads | --rust-client 模式下解析响应流的线程数；默认为可用并行度 | — | — |

#### 3.1.2 请求生成与速率控制

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --num-prompts | 要处理的 prompt 数量 | 1000 | — |
| --request-rate | 每秒请求数；为 inf 时所有请求在 0 时刻一次性全部发出，否则用 Poisson 过程合成请求到达时间 | inf | — |
| --max-concurrency | 最大并发请求数，用于模拟上层组件限制并发；与 --request-rate 组合时，若服务器处理不过来，实际请求率可能低于设定值 | — | — |
| --seed | 随机种子 | 42 | — |
| --warmup-requests | 正式测试前运行的 warmup 请求数 | 1 | — |

#### 3.1.3 数据集通用

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --dataset-name | 基准测试使用的数据集名称 | sharegpt | agentic-trace / sharegpt / custom / openai / random / random-ids / generated-shared-prefix / mmmu / image / mooncake / longbench_v2 / speed-bench |
| --dataset-path | 数据集文件路径 | "" | — |

#### 3.1.4 sharegpt 数据集

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --sharegpt-output-len | 每个请求的输出长度；覆盖 ShareGPT 数据集自带的输出长度 | — | — |
| --sharegpt-context-len | 模型上下文长度（ShareGPT 数据集）；超过该长度的请求将被丢弃 | — | — |

#### 3.1.5 random 数据集

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --random-input-len | 每个请求的输入 token 数，仅用于 random 与 image 数据集 | 1024 | — |
| --random-output-len | 每个请求的输出 token 数，仅用于 random 与 image 数据集 | 1024 | — |
| --random-range-ratio | 输入/输出长度的采样波动比例范围，仅用于 random 与 image 数据集 | 0.0 | — |

#### 3.1.6 generated-shared-prefix 数据集

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --gsp-num-groups | 系统 prompt 分组数（generated-shared-prefix 数据集） | 64 | — |
| --gsp-prompts-per-group | 每个系统 prompt 组内的 prompt 数 | 16 | — |
| --gsp-system-prompt-len | 系统 prompt 的目标 token 长度 | 2048 | — |
| --gsp-question-len | 问题（question）的目标 token 长度 | 128 | — |
| --gsp-output-len | 输出的目标 token 长度 | 256 | — |
| --gsp-range-ratio | 输入/输出长度的采样波动比例范围（仅 gsp 数据集）；注意默认 1.0 为向后兼容而设，与 random 数据集默认 0.0 不同 | 1.0 | — |
| --gsp-fast-prepare | 通过去除统计计算加速数据准备，会使部分输出统计不准确，但适合压力测试 | False | — |
| --gsp-send-routing-key | 通过 X-SMG-Routing-Key 请求头携带路由键；相同前缀的请求共享同一路由键 | False | — |
| --gsp-num-turns | 多轮对话的轮数；大于 1 时每个 prompt 变为共享同一系统前缀的问题列表 | 1 | — |
| --gsp-ordered | 保持请求顺序不打乱；默认将请求随机打乱 | False | — |
| --gsp-group-distribution | 前缀组的采样分布；uniform 为每组均分请求数，zipf 按 p(rank) = (1/rank^alpha) / sum_k(1/k^alpha) 采样（rank 从 1 开始，组索引 0 最热）；总请求数仍为 组数 x 每组 prompt 数，仅改变每请求的组分配；磁盘数据集缓存按 (group_distribution, zipf_alpha) 区分键 | uniform | uniform / zipf |
| --gsp-zipf-alpha | --gsp-group-distribution=zipf 的 Zipf 指数 alpha；必须为严格大于 0 的有限浮点数，值越大请求越集中于低 rank（更热）的组；设为 zipf 时必须提供 | — | — |

#### 3.1.7 image / mmmu 多模态数据集

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --image-count | 每个请求的图片数量（仅 image 数据集） | 1 | — |
| --image-resolution | 图片分辨率（image 数据集）；支持预设 4k/1080p/720p/360p、自定义"高x宽"（如 1080x1920）或随机范围 random:最小高x宽-最大高x宽（如 random:256x256-1024x1024） | 1080p | — |
| --random-image-count | 启用随机图片数量 | False | — |
| --image-format | 图片格式（image 数据集），支持 jpeg 和 png | jpeg | jpeg / png |
| --image-content | 图片内容（image 数据集），random 为随机噪声图，blank 为空白图 | random | random / blank |

#### 3.1.8 speed-bench 数据集

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --speed-bench-category | speed-bench 数据集的类别过滤 | — | low_entropy / mixed / high_entropy |
| --speed-bench-output-len | speed-bench 请求的固定输出长度 | 512 | — |

#### 3.1.9 mooncake / agentic-trace 数据集

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --dataset-offset | 采样前将对话列表轮转该条数（agentic-trace 数据集），使连续 sweep 步骤从新对话开始 | 0 | — |
| --agentic-max-turns | 将每个对话限制到至多该轮数（agentic-trace 数据集）；默认使用 trace 中的全部轮数 | — | — |
| --use-trace-timestamps | 使用 trace 文件中的时间戳调度请求（仅对 mooncake 数据集有效） | False | — |
| --mooncake-slowdown-factor | mooncake trace 重放的减速因子，2.0 表示重放速度放慢 2 倍；mooncake 模式下 --request-rate 被忽略 | 1.0 | — |
| --mooncake-num-rounds | mooncake 数据集每个会话的对话轮数；大于 1 时启用真正的多轮会话测试 | 1 | — |
| --mooncake-workload | mooncake 数据集的底层负载类型 | conversation | mooncake / conversation / synthetic / toolagent |

#### 3.1.10 采样与请求体

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --temperature | 采样温度 | 0.0 | — |
| --top-p | 核采样（nucleus sampling）参数 | 1.0 | — |
| --extra-request-body | 将给定 JSON 对象附加到请求体（如 {"key1": "value1", "key2": "value2"}），可用于指定采样参数等额外生成参数 | — | — |
| --apply-chat-template | 对输入应用 chat 模板 | False | — |
| --prompt-suffix | 附加到所有用户 prompt 末尾的后缀，其后再加上 assistant prompt 后缀 | "" | — |
| --tokenize-prompt | 输入使用整数 token id 而非字符串，便于精确控制 prompt 长度（目前仅兼容 --backend sglang） | False | — |
| --return-logprob | 返回 logprob | False | — |
| --top-logprobs-num | 每个 token 返回的 top logprobs 数量，仅与 --return-logprob 配合使用 | 0 | — |
| --token-ids-logprob | 需探测 logprob 的 token ID 列表，如 --token-ids-logprob 1 2 10 100 1000，仅与 --return-logprob 配合使用 | — | — |
| --logprob-start-len | 返回输入 logprob 的起始位置；-1 表示不返回输入 logprob，0 表示全部返回，仅与 --return-logprob 配合使用 | -1 | — |
| --return-routed-experts | 返回路由到的专家（用于 MoE 模型） | False | — |
| --disable-ignore-eos | 禁用忽略 EOS（默认忽略 EOS 强制生成满输出长度；设置本项后请求在 EOS 处停止） | False | — |
| --disable-stream | 禁用流式模式 | False | — |
| --lora-name | LoRA 适配器名称，可提供多个，格式为 {name} {name} {name}... | — | — |
| --lora-request-distribution | --lora-name 所列 LoRA 适配器的采样分布（借鉴 Punica 论文）；distinct 表示每个请求选择一个新 LoRA，skewed 服从 Zipf 分布（模型 i 的请求数是模型 i+1 的 alpha 倍，alpha > 1） | uniform | uniform / distinct / skewed |
| --lora-zipf-alpha | --lora-request-distribution=skewed 时 Zipf 分布的参数 | 1.5 | — |
| --pd-separated | 基准测试 PD 分离（prefill-decode 分离）服务器 | False | — |
| --fake-prefill | 启用 fake prefill 模式进行纯 decode 测试；配合运行 --disaggregation-transfer-backend fake 的 decode 服务器，可在没有真实 prefill 节点的情况下测试纯 decode 性能（自动在请求体注入 bootstrap_host / bootstrap_room 字段） | False | — |

#### 3.1.11 日志与输出

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --output-file | 输出 JSONL 文件名 | — | — |
| --output-details | 输出基准测试的详细信息 | False | — |
| --print-requests | 测试期间即时打印请求内容，便于快速发现问题 | False | — |
| --disable-tqdm | 禁用 tqdm 进度条 | False | — |
| --plot-throughput | 绘制吞吐与并发请求数随时间变化的曲线，需安装 termplotlib 和 gnuplot | False | — |
| --tag | 写入输出结果中的标签，用于区分不同测试 | — | — |

#### 3.1.12 性能分析与缓存

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --profile | 使用 Torch Profiler；服务端必须以 SGLANG_TORCH_PROFILER_DIR 环境变量启动才能启用分析器 | False | — |
| --profile-activities | 分析器捕获的活动类型；MEM 会 dump 一份 torch.cuda 内存快照（可在 pytorch.org/memory_viz 查看） | ["CPU", "GPU"] | CPU / GPU / CUDA_PROFILER / XPU / MEM |
| --profile-start-step | 经过该数量个 forward 步后再开始分析，可用于 warmup | — | — |
| --profile-steps | 分析的步数；指定后经过该步数后自动停止分析 | — | — |
| --profile-num-steps | 分析的 forward 步数（作为 start_profile 接口的 num_steps 参数）；启用 --profile-by-stage 时若未指定则默认取 5 | — | — |
| --profile-by-stage | 按阶段（prefill / decode）进行分析 | False | — |
| --profile-stages | 配合 --profile-by-stage 指定要分析的阶段列表（如 prefill decode） | — | — |
| --profile-output-dir | 分析 trace 的输出目录（默认取 SGLANG_TORCH_PROFILER_DIR 环境变量，未设置时为 /tmp） | — | — |
| --profile-prefix | 分析 trace 文件名的前缀 | — | — |
| --profile-prefill-url | PD 分离模式下用于分析的 prefill worker URL，可指定多个；与 --profile-decode-url 互斥，PD 分离模式下 prefill 与 decode worker 必须分开分析 | — | — |
| --profile-decode-url | PD 分离模式下用于分析的 decode worker URL，可指定多个；与 --profile-prefill-url 互斥，PD 分离模式下 prefill 与 decode worker 必须分开分析 | — | — |
| --flush-cache | 运行基准测试前刷新服务端 cache | False | — |
| --flush-cache-timeout | 等待 SGLang 服务器空闲以刷新 cache 的最大秒数 | 60.0 | — |
| --cache-report | 测试后收集并显示 cache 命中统计（仅 sglang 系后端 sglang / sglang-native / sglang-oai / sglang-oai-chat 支持） | False | — |

### 3.2 sglang.bench_one_batch（单批延迟基准测试）

`sglang.bench_one_batch`（旧路径 `python3 -m sglang.bench_one_batch`，新路径 `python3 -m sglang.benchmark.one_batch`）是无服务器的单批静态延迟测试工具：它不启动 HTTP 服务器，直接使用 SGLang 底层 API（Engine）构造单个批次并同步执行 forward，测量单批 prefill 延迟、decode 延迟及对应吞吐。支持对多组 (batch-size, input-len, output-len) 组合做笛卡尔积扫描，也支持正确性测试（截断 prefill 后再 extend，验证 KV cache 一致性）与 Torch profiler 性能分析。

典型用法示例：

```bash
# 使用 dummy 权重快速测试单批 prefill/decode 延迟（跳过实际权重加载）
python3 -m sglang.bench_one_batch \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --load-format dummy \
    --batch-size 16 --input-len 4096 --output-len 256

# 多 (batch, input, output) 组合笛卡尔积扫描，结果追加写入 jsonl
python3 -m sglang.bench_one_batch \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tp 2 \
    --batch-size 1 4 16 64 \
    --input-len 128 2048 32768 \
    --output-len 32 256 \
    --run-name llama3-8b-tp2 \
    --result-filename bench_results.jsonl
```

BenchArgs 专属参数如下：

| 参数 | 说明 | 默认值 | 可选值 |
|---|---|---|---|
| --run-name | 本次基准测试运行的名称，用于结果记录 | default | — |
| --batch-size | 批大小；支持多个值（nargs+），会对所有 (batch-size, input-len, output-len) 组合做笛卡尔积扫描 | 1 | — |
| --input-len | 输入长度；支持多个值 | 1024 | — |
| --output-len | 输出长度；支持多个值 | 16 | — |
| --prompt-filename | 从文件读取自定义 prompt（每行一个），替代随机 token | "" | — |
| --result-filename | 结果 jsonl 输出文件（追加模式） | result.jsonl | — |
| --correctness-test | 运行正确性测试模式而非延迟测试（截断 prefill 再 extend，验证 KV cache 正确性） | False | — |
| --cut-len | 仅用于正确性测试：首次 prefill 的截断长度 | 4 | — |
| --log-decode-step | 按解码步记录延迟日志，0 表示禁用 | 0 | — |
| --profile | 启用性能分析 | False | — |
| --profile-record-shapes | 分析结果中记录张量形状 | False | — |
| --profile-activities | 分析器活动类型 | ["CPU", "GPU"] | CPU / GPU / XPU / CUDA_PROFILER |
| --profile-stage | 分析哪个阶段 | all | all / prefill / decode |
| --profile-prefix | 分析结果文件名前缀，完整文件名为 [prefix]_batch[bs]_input[il]_output[ol].trace.json.gz | profile | — |
| --profile-start-step | 从第几个解码步开始分析（0-indexed） | None（默认 output_len//2） | — |
| --profile-steps | 从起始步开始分析多少个解码步 | None（默认 1 步） | — |

此外，该命令还接受与 launch_server 完全相同的全部 ServerArgs（如 --model-path、--tp、--mem-fraction-static、--load-format 等），详见《vLLM_SGLang_serve.md》第三章。
