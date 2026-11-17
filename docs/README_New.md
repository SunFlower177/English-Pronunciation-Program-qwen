# Evidence-Grounded CAPT：英语照稿朗读发音诊断系统

本项目实现模式 A：**照稿朗读模式**。用户输入英文原文，上传或录制一段朗读音频，系统输出：

1. ASR 转写文本；
2. 原文与 ASR 的词级对齐；
3. 单词级错误检测；
4. 音素级错误检测；
5. 停顿检测；
6. 连读检测；
7. 重音和语调检测；
8. evidence-grounded LLM 反馈，即先产生结构化证据，再由 Qwen2-Audio 或模板生成反馈。

系统没有把“判错”交给 LLM 猜，而是把判错放在 `diagnostics.py` 中，由 ASR、强制对齐、音素识别、VAD、pitch/intensity 等证据产生；LLM 只负责把证据转成学生能看懂的解释。

---

## 1. 七个功能模块

| 模块 | 文件 | 主要职责 |
|---|---|---|
| 1 | `capt_pronunciation/audio_io.py` | 音频转码、16kHz mono wav、轻量 VAD、pitch/intensity 特征 |
| 2 | `capt_pronunciation/transcription.py` | WhisperX ASR、WhisperX 词级时间戳、faster-whisper fallback |
| 3 | `capt_pronunciation/reference_alignment.py` | 原文规范化、CMUdict/G2P、参考词与 ASR 词对齐、MFA 强制对齐 |
| 4 | `capt_pronunciation/diagnostics.py` | 单词、音素、停顿、连读、重音、语调错误证据检测 |
| 5 | `capt_pronunciation/feedback.py` | Qwen2-Audio 或模板反馈生成，严格基于 evidence JSON |
| 6 | `capt_pronunciation/runtime.py` | 端到端 pipeline、CLI、FastAPI 接口 |
| 7 | `capt_pronunciation/evaluation.py` | WER、PER、MDD precision/recall/F1、反馈人工评分聚合 |

辅助文件：

- `capt_pronunciation/contracts.py`：Pydantic 数据结构和配置，不算功能模块；
- `web/index.html`：最小前端，支持上传和浏览器录音；
- `docs/ARCHITECTURE.md`：技术架构；
- `docs/CODEX_GUIDE.md`：给 Codex/后续开发者的修改指南。

---

## 2. 安装

### 2.1 推荐环境

```bash
conda create -n capt python=3.10 -y
conda activate capt
```

安装系统依赖：

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y ffmpeg espeak-ng
```

安装 Python 依赖：

```bash
cd evidence_capt_project
pip install -e ".[all]"
```

如果 Qwen2-Audio 报 `KeyError: qwen2-audio`，根据 Qwen2-Audio 官方说明改用 transformers 源码版：

```bash
pip install git+https://github.com/huggingface/transformers
```

### 2.2 MFA 安装

MFA 推荐用 conda-forge 安装：

```bash
conda install -c conda-forge montreal-forced-aligner -y
bash scripts/setup_mfa_models.sh
```

然后在 `.env` 或环境变量中启用：

```bash
export CAPT_USE_MFA=1
```

MFA 没有安装时系统仍能运行，但音素边界会使用词内等分估计，论文实验中建议启用 MFA。

---

## 3. 运行

### 3.1 命令行

```bash
capt-analyze \
  --audio path/to/reading.wav \
  --reference examples/reference.txt \
  --analysis-id demo001 \
  --verbose
```

输出 JSON，同时保存到：

```text
runs/demo001/report.json
runs/demo001/asr.json
runs/demo001/alignment.json
runs/demo001/evidence.json
```

### 3.2 Web API + 前端

```bash
uvicorn capt_pronunciation.runtime:app --host 0.0.0.0 --port 8000
```

浏览器打开：

```text
http://localhost:8000
```

API：

```bash
curl -X POST http://localhost:8000/api/analyze-reading \
  -F "reference_text=I want to improve my English pronunciation." \
  -F "audio=@reading.wav"
```

---

## 4. 关键环境变量

复制 `.env.example`，按机器配置修改。

```bash
CAPT_DEVICE=cpu                  # cuda 或 cpu
CAPT_COMPUTE_TYPE=int8           # CPU 建议 int8；GPU 可 float16
CAPT_WHISPERX_MODEL=small.en     # tiny.en/base.en/small.en/medium.en/large-v3 等
CAPT_USE_MFA=0                   # 装好 MFA 后改成 1
CAPT_USE_PHONEME_MODEL=1
CAPT_PHONEME_MODEL=facebook/wav2vec2-xlsr-53-espeak-cv-ft
CAPT_USE_QWEN2_AUDIO=0           # 默认关闭；7B 模型很重
CAPT_QWEN2_AUDIO_MODEL=Qwen/Qwen2-Audio-7B-Instruct
```

建议开发阶段：

```bash
CAPT_DEVICE=cpu
CAPT_COMPUTE_TYPE=int8
CAPT_WHISPERX_MODEL=tiny.en
CAPT_USE_MFA=0
CAPT_USE_QWEN2_AUDIO=0
```

建议论文实验：

```bash
CAPT_DEVICE=cuda
CAPT_COMPUTE_TYPE=float16
CAPT_WHISPERX_MODEL=large-v3
CAPT_USE_MFA=1
CAPT_USE_PHONEME_MODEL=1
CAPT_USE_QWEN2_AUDIO=1
```

---

## 5. 输出 JSON 结构

顶层：

```json
{
  "analysis_id": "demo001",
  "mode": "reading",
  "reference_text": "...",
  "audio": {...},
  "asr": {...},
  "alignment": {...},
  "evidence": {...},
  "feedback": {...}
}
```

其中 `evidence` 是论文最重要的可解释中间结果：

```json
{
  "word_issues": [],
  "phoneme_issues": [],
  "pause_issues": [],
  "linking_issues": [],
  "stress_issues": [],
  "intonation_issues": [],
  "phoneme_hypotheses": [],
  "prosody": {}
}
```

每个 issue 都有：

```json
{
  "issue_id": "I0001",
  "type": "phoneme_substitution",
  "severity": "medium",
  "ref_index": 3,
  "word": "think",
  "observed": "s",
  "expected": "θ",
  "start": 1.23,
  "end": 1.56,
  "confidence": 0.72,
  "evidence": {...},
  "suggestion_hint": "..."
}
```

---

## 6. 论文实验建议

最低限度做三组对比：

1. `WhisperX + LLM direct feedback`：只给 ASR 文本和原文，让 LLM 评价；
2. `Speech LLM direct`：把音频和原文直接给 Qwen2-Audio，让它评价；
3. `Ours`：本项目 evidence-grounded pipeline。

指标：

- ASR：WER；
- Word-level MDD：precision / recall / F1；
- Phoneme-level MDD：precision / recall / F1；
- 评分任务：Pearson / Spearman / MAE；
- 反馈质量：正确性、具体性、可操作性、幻觉率、学习者满意度。

`capt_pronunciation/evaluation.py` 已经提供基础评测函数。

---

## 7. 当前实现边界

- 模式 A 已实现；模式 B 在 `CaptPipeline.analyze_free_speaking()` 中保留扩展点，但没有启用；
- 音素级诊断依赖 Wav2Vec2/CTC phoneme checkpoint，实际性能需要在 L2-ARCTIC、Speechocean762 或自建数据上调阈值；
- 连读、停顿、重音、语调使用启发式规则，适合作为可解释 evidence baseline；
- Qwen2-Audio 默认关闭，因为 7B 模型需要较多显存；关闭时系统使用确定性模板反馈，便于复现实验。
