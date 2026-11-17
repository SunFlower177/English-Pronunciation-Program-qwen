# ModeA / ModeB / ModeC 确定性改造与 PCC 运行说明

本流程的评分源是结构化 evidence，不调用 Qwen，也不读取待评价样本的人工分数。
人工分数只在最后一步用于计算 PCC、Spearman、MAE、RMSE 和 bootstrap 95% CI。

## 1. 直接重算现有 ModeA 报告

    .\.venv\Scripts\python.exe scripts\evaluate_deterministic_pcc.py --run-config configs\deterministic_pcc_modeA.yaml

输出：

- runs/deterministic_modeA_pcc_v2/deterministic_predictions_modeA.csv
- runs/deterministic_modeA_pcc_v2/deterministic_pcc_modeA.json

默认参数只重评分已有 report.json，不会重新跑 ASR、音素模型或声学提取。
修改 configs/deterministic_scoring.yaml 后可用同一命令立即重算 PCC。

现有 MultiPA reference text 是 ASR-assisted，因此该结果是诊断实验，不是严格的
scripted-reading 论文结果。正式 ModeA 应把 annotation、wav_dir、reference_map
换成带真实 target text 的数据。

## 2. 修改模型或 evidence 后重新生成报告

在 configs/deterministic_pcc_modeA.yaml 中设置：

    run_analysis: true
    force_analysis: true

随后运行同一命令。批处理会强制 use_qwen_text=false 和
use_qwen2_audio=false。

ModeA 连续 evidence 包括：

- reference/ASR 全局单词相似度和插入、删除、替换；
- 同作用域 canonical/learner 全局音素编辑相似度；
- 非 blank CTC 发射置信度和 canonical CTC-GOP；
- 音素、词时间、pitch、intensity coverage；
- speech rate、articulation rate、pause ratio、MLR；
- F0 range、nPVI、duration CV、stress/intonation evidence。

## 3. ModeB 参考映射与多参考

复制 configs/modeB_reference_map.example.csv，每行填写：

    audio,reference_text,reference_audio
    learner_001.wav,Pick it up.,references/ref_a.wav|references/ref_b.wav|references/ref_c.wav

多个参考录音使用 | 或 ; 分隔。然后更新
configs/deterministic_pcc_modeB.yaml 中的 reference_map 和 report_dir，
并设置 run_analysis: true。

    .\.venv\Scripts\python.exe scripts\evaluate_deterministic_pcc.py --run-config configs\deterministic_pcc_modeB.yaml

ModeB 会保留连续的 phone/reference similarity、pause MAE、relative-duration MAE、
tempo log-ratio、F0 DTW、F0 correlation、word/sentence stress overlap。多个参考的
最终分数默认取 median。

可选 WavLM Soft-DTW：

    $env:CAPT_IMITATION_USE_SSL="1"
    $env:CAPT_IMITATION_SSL_MODEL="microsoft/wavlm-base-plus"
    $env:CAPT_IMITATION_SSL_LAYER="9"

模型不可用时 SSL evidence 为 missing，评分器按可用权重重归一化，不会伪造满分。

## 4. 参数冻结规则

configs/deterministic_scoring.yaml 是唯一评分参数源。可以在开发集上调整：

- 各 evidence 权重；
- speech/articulation rate 的 soft/ideal 区间；
- pause、pitch、nPVI、duration 的映射；
- ModeB 各距离的指数衰减 scale。

正式实验前必须冻结文件并记录 score_config_version。不要在最终测试集上依据
PCC 反复调参。PCC 对单一分数的正线性缩放不敏感，真正有用的是 evidence 融合
改变样本排序。

## 5. ModeC 真实 parser

安装可选依赖及至少一个训练解析模型：

    .\.venv\Scripts\python.exe -m pip install -e ".[syntax]"
    .\.venv\Scripts\python.exe -m spacy download en_core_web_trf
    .\.venv\Scripts\python.exe -c "import stanza; stanza.download('en')"

也可准备 Stanza English models。运行时优先级：

1. spaCy dependency + Stanza constituency；
2. Stanza dependency + constituency；
3. heuristic 低置信度降级。

成分树现在会转换为结构化 S/SBAR/VP/NP spans，并与 dependency clause spans
进行一致性校验。that 会根据 dependency、紧邻先行词和 complement-taking
verb 区分关系从句与名词性补语从句；不定式要求 to + VERB；强调句也会验证
that 后存在谓语。semantic role 输出已改为 dependency-grounded argument map，
不再把从句类型直接冒充 PropBank SRL。
