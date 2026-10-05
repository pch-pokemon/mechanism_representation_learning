# Resolving hidden oxidation states beyond kinetic responses in superalloys through large-language-model-enabled mechanism representation learning

This repository contains the code, prompt templates, optimization scripts, and processed source data supporting the study:

**“Resolving hidden oxidation states beyond kinetic responses in superalloys through large-language-model-enabled mechanism representation learning.”**

The framework organizes literature-derived oxidation knowledge into six physically defined mechanism channels, constructs continuous channel-specific mechanism representations, learns these representations from structured material variables, transfers the pretrained representation to oxidation-kinetics prediction, and uses training-reference neighbourhoods for literature-grounded interpretation of predicted oxidation states.

---

## Repository structure

```text
.
├── README.md
├── optimization/
│   ├── optimize_pretraining.py
│   └── optimize_frozen_transfer.py
├── prompt/
│   ├── extraction_channel1-6_prompt.json
│   └── retrieval_interpretation_prompts.json
├── source_data/
│   ├── Fig.2/
│   ├── Fig.3/
│   ├── Fig.4/
│   ├── Fig.5/
│   ├── Fig.6/
│   ├── Fig.7/
│   ├── Table1-2/
│   ├── Table4/
│   └── SM/
│       ├── Fig.S2-Fig.S7/
│       ├── Fig.S8/
│       ├── Fig.S9_TableS4/
│       ├── Fig.S10_TableS5/
│       ├── TableS2/
│       └── TableS3/
├── analyze_mechanism_cases.py
├── build_mechanism_embeddings.py
├── evaluate_kinetics_release.py
├── evaluate_pretraining.py
├── pretrain_mechanism_encoder.py
├── screen_mechanism_divergence_release.py
└── train_frozen_transfer.py
```

---

## Framework overview

The computational workflow contains five main stages.

1. **Mechanism knowledge organization**  
   Literature-derived sample-condition-specific oxidation evidence is organized into six physically complementary channels:
   - protective-scale formation;
   - internal and selective oxidation;
   - non-protective or transient oxide formation;
   - transport and defect kinetics;
   - scale failure and spallation;
   - alloying-element interactions.

2. **Reference mechanism representation construction**  
   Channel-specific textual mechanism descriptions are encoded with SciBERT. Final-layer token representations are mean-pooled and L2-normalized, followed by channel-wise PCA. PCA is fitted using training-set mechanism descriptions only and applied to validation and test records without refitting. A common dimensionality of 64 is used for all six channels.

3. **Mechanism-supervised pretraining**  
   Structured material variables are mapped to a shared latent representation, from which six independent output heads predict the six 64-dimensional mechanism representations. Only sample-channel combinations with reported mechanism evidence contribute to the channel-wise cosine-distance loss.

4. **Frozen transfer to oxidation kinetics**  
   The pretrained shared encoder and six mechanism heads are frozen. A kinetics prediction module is trained on top of the frozen latent representation to predict `lg(kp)`. The six mechanism vectors remain parallel outputs and are not concatenated into the kinetics head.

5. **Mechanism-space interpretation and prospective screening**  
   Predicted mechanism vectors are interpreted through their Top-5 nearest neighbours in the **training-reference mechanism library** using cosine similarity. Held-out original mechanism descriptions are excluded from interpretation generation and used only for post-hoc validation. The same representation framework is further used to screen compositionally distinct candidate alloys with similar predicted kinetics but divergent multichannel mechanism states.

---

## Main scripts

### `build_mechanism_embeddings.py`

Constructs the six channel-specific reference mechanism spaces.

Main operations include:

- text-only SciBERT semantic encoding;
- mean pooling of final-layer token representations;
- L2 normalization;
- training-only PCA fitting;
- transformation of validation/test embeddings without refitting;
- export of 64-dimensional channel-specific reference vectors;
- PCA explained-variance diagnostics.

Canonical mechanism tags are retained for annotation and interpretation but are not included in the semantic embedding input.

### `pretrain_mechanism_encoder.py`

Trains the six-channel mechanism-supervised encoder.

Key features:

- shared structured-input encoder;
- six independent 64-dimensional mechanism heads;
- channel-specific masking for `not_reported` states;
- cosine-distance supervision;
- fixed train/validation/test split;
- training-set feature normalization;
- validation-based checkpoint selection;
- held-out test set excluded from model selection.

### `evaluate_pretraining.py`

Evaluates the final pretrained mechanism encoder on the held-out test subset.

The script reports three complementary levels of representation recovery:

- pointwise alignment between predicted and reference mechanism vectors;
- pairwise similarity preservation within each mechanism space;
- reference-state retrieval of the corresponding held-out mechanism state.

Reported metrics include cosine-similarity statistics, Spearman correlation, mean reciprocal rank, median reference rank, Recall@3, and Recall@5.

This script supports Fig. 3, Fig. 4, Table 1, and Table 2.

### `train_frozen_transfer.py`

Performs frozen transfer from the pretrained mechanism encoder to oxidation-kinetics prediction.

During transfer:

- the shared pretrained encoder is frozen;
- all six mechanism heads are frozen;
- only the kinetics prediction module is optimized;
- the kinetics head receives the frozen latent material representation;
- the six mechanism vectors remain parallel model outputs;
- optimization uses mean squared error on `lg(kp)`;
- validation RMSE is used for checkpoint selection;
- the test set is not used during training or model selection.

### `evaluate_kinetics_release.py`

Performs the final held-out evaluation of the frozen-transfer kinetics model.

The script reports:

- coefficient of determination (`R²`);
- mean absolute error (`MAE`);
- root mean squared error (`RMSE`).

It also exports held-out predictions and predicted six-channel mechanism vectors for downstream interpretation.

This script supports Fig. 5 and subsequent mechanism-case analyses.

### `analyze_mechanism_cases.py`

Performs mechanism-space interpretation of selected held-out test cases.

For each selected sample and each mechanism channel:

1. the predicted 64-dimensional mechanism vector is used as the query;
2. the Top-5 nearest mechanism states are retrieved from the **training-reference library**;
3. cosine similarity to the held-out original mechanism vector is calculated separately for post-hoc validation;
4. the predicted state, held-out original state, Top-5 neighbours, and training-reference library are projected into a fixed channel-specific UMAP mapping.

The held-out original mechanism is not used during retrieval-grounded interpretation.

This script supports Fig. 6, Fig. S9, Table S3, and Table S4.

### `screen_mechanism_divergence_release.py`

Performs prospective mechanism-aware screening of Co-based candidate alloy states.

The search space varies:

- Al: 7–15 at.%;
- Cr: 3–15 at.%;
- Ni: 20–35 at.%;
- W: 0–8 at.%;
- Ti: 0–8 at.%;
- Mo: 0–4 at.%;
- Ta: 0–6 at.%;
- Si: 0–3 at.%;

with Co treated as the balance element.

The workflow applies Latin hypercube sampling, 0.5 at.% discretization, feasibility and duplicate filtering, frozen-transfer prediction of `lg(kp)` and six mechanism representations, pairwise kinetic pre-screening, and channel-wise normalization of cosine distance against the cross-source training-reference distance distribution.

The effective number of contributing channels is defined as

\[N_{\mathrm{eff}}=\frac{\left(\sum_c q_c\right)^2}{\sum_c q_c^2},\]

and the balanced multichannel mechanism-divergence score is

\[D_{\mathrm{mech}}=\bar{q}\sqrt{\frac{N_{\mathrm{eff}}}{6}}.\]

The final target region is defined by

\[|\Delta \mathrm{lg}(k_p)| \leq 0.05\]

and

\[D_{\mathrm{mech}} \geq 0.60.\]

No additional hidden pair-selection criteria are used in the release implementation.

This script supports Fig. 7, Fig. S10, and Table S5.

---

## Hyperparameter optimization

The `optimization/` directory contains the Optuna workflows used to select model hyperparameters from the validation subset.

### `optimization/optimize_pretraining.py`

Searches:

- network depth;
- hidden-layer widths;
- dropout;
- learning rate;
- weight decay.

The Optuna objective is the **mean validation cosine similarity across the six mechanism channels**. The held-out test set is excluded from trial ranking, pruning, checkpoint selection, and hyperparameter selection.

### `optimization/optimize_frozen_transfer.py`

Searches the kinetics-head:

- network depth;
- hidden-layer widths;
- dropout;
- learning rate;
- weight decay.

The pretrained encoder and six mechanism heads remain frozen throughout every trial. The Optuna objective is **validation RMSE**, and the held-out test set is excluded from hyperparameter optimization.

---

## Prompt templates

The `prompt/` directory contains the LLM prompts used in the mechanism extraction and interpretation workflow.

### `extraction_channel1-6_prompt.json`

Defines the joint six-channel mechanism-extraction framework, including:

- channel definitions;
- allowed canonical tags;
- evidence-status definitions;
- sample-condition alignment rules;
- channel boundaries;
- cross-channel consistency checks;
- prohibited unsupported inferences;
- required structured JSON output.

The extracted mechanism descriptions form the textual basis for the reference mechanism representations.

### `retrieval_interpretation_prompts.json`

Contains the downstream prompts used for:

- channel-wise interpretation from Top-5 training-reference neighbours;
- integration of six channel-level interpretations into an overall oxidation-state description;
- post-hoc comparison between predicted and held-out literature mechanisms;
- assignment of `Consistent`, `Partially consistent`, or `Inconsistent` agreement labels;
- retrieval-grounded comparison of the representative prospective candidate pair.

For held-out case analysis, interpretation and validation are intentionally separated: held-out original mechanism descriptions are not provided during retrieval-grounded interpretation generation.

---

## Source data

The `source_data/` directory contains the processed numerical and tabular data supporting the figures and tables released with this repository.

### Main-text source data

```text
source_data/
├── Fig.2/
├── Fig.3/
├── Fig.4/
├── Fig.5/
├── Fig.6/
├── Fig.7/
├── Table1-2/
└── Table4/
```

### Supplementary source data

```text
source_data/SM/
├── Fig.S2-Fig.S7/
├── Fig.S8/
├── Fig.S9_TableS4/
├── Fig.S10_TableS5/
├── TableS2/
└── TableS3/
```

The released source-data folders contain the processed values used to reproduce or audit the corresponding figures and tables.

---

## Recommended execution order

```text
1. build_mechanism_embeddings.py
        ↓
2. optimization/optimize_pretraining.py
        ↓
3. pretrain_mechanism_encoder.py
        ↓
4. evaluate_pretraining.py
        ↓
5. optimization/optimize_frozen_transfer.py
        ↓
6. train_frozen_transfer.py
        ↓
7. evaluate_kinetics_release.py
        ↓
8. analyze_mechanism_cases.py
        ↓
9. screen_mechanism_divergence_release.py
```

---

## Software environment

The SciBERT encoder used for construction of the literature-derived semantic reference representations is:

```text
allenai/scibert_scivocab_uncased
```

---

## Data availability

The processed data supporting the released figures, tables, and principal quantitative analyses are provided in `source_data/`.

The complete literature-derived dataset used for model development, including the full sample-condition-level structured dataset and full mechanism corpus, is **not included in this repository because it remains part of ongoing research**.

Consequently, this repository supports inspection and reproduction of the released figure/table source data and documents the complete modeling workflow, whereas full end-to-end model retraining requires access to the underlying literature-derived development dataset.

---------------------------
# Resolving hidden oxidation states beyond kinetic responses in superalloys through large-language-model-enabled mechanism representation learning

本仓库包含以下研究所使用的代码、Prompt 模板、超参数优化脚本以及已处理的图表源数据：

**“Resolving hidden oxidation states beyond kinetic responses in superalloys through large-language-model-enabled mechanism representation learning.”**

该框架将文献中的氧化机理知识组织为六个具有明确物理含义的机理通道，构建连续的通道特异性机理表征，并从结构化材料变量中学习这些表征；随后将预训练得到的表征冻结迁移至氧化动力学预测任务，并通过训练集参考机理邻域对预测氧化状态进行文献可追溯的解释。

---

## 仓库结构

```text
.
├── README.md
├── optimization/
│   ├── optimize_pretraining.py
│   └── optimize_frozen_transfer.py
├── prompt/
│   ├── extraction_channel1-6_prompt.json
│   └── retrieval_interpretation_prompts.json
├── source_data/
│   ├── Fig.2/
│   ├── Fig.3/
│   ├── Fig.4/
│   ├── Fig.5/
│   ├── Fig.6/
│   ├── Fig.7/
│   ├── Table1-2/
│   ├── Table4/
│   └── SM/
│       ├── Fig.S2-Fig.S7/
│       ├── Fig.S8/
│       ├── Fig.S9_TableS4/
│       ├── Fig.S10_TableS5/
│       ├── TableS2/
│       └── TableS3/
├── analyze_mechanism_cases.py
├── build_mechanism_embeddings.py
├── evaluate_kinetics_release.py
├── evaluate_pretraining.py
├── pretrain_mechanism_encoder.py
├── screen_mechanism_divergence_release.py
└── train_frozen_transfer.py
```

---

## 框架概述

整个计算流程包括五个主要阶段。

1. **机理知识组织**  
   将文献中样本-条件级别的氧化证据组织为六个物理上互补的机理通道：
   - 保护性氧化膜形成；
   - 内氧化与选择性氧化；
   - 非保护性或瞬态氧化物形成；
   - 传输与缺陷动力学；
   - 氧化膜失效与剥落；
   - 合金元素相互作用。

2. **参考机理表征构建**  
   使用 SciBERT 对各通道的机理文本进行语义编码。最终层 token 表征经均值池化和 L2 归一化后，对每个机理通道分别进行 PCA。PCA 仅使用训练集机理文本对应的嵌入进行拟合，随后在不重新拟合的情况下应用于验证集和测试集。六个通道统一采用 64 维参考表征。

3. **机理监督预训练**  
   结构化材料变量首先映射为共享潜在表征，再由六个独立输出头分别预测六个 64 维机理表征。只有具有有效机理证据的样本-通道组合参与通道级 cosine-distance 损失计算。

4. **冻结迁移至氧化动力学预测**  
   预训练得到的共享编码器和六个机理头全部冻结，仅训练新的动力学预测模块以预测 `lg(kp)`。六个机理向量保留为并行输出，不直接拼接为动力学预测头的输入。

5. **机理空间解释与前瞻筛选**  
   预测机理向量通过 cosine similarity 在**训练集参考机理库**中检索 Top-5 最近邻，用于生成文献可追溯的机理解释。held-out 原始机理描述不会用于解释生成，仅用于后验验证。相同的机理表征框架进一步用于筛选预测动力学相近但多通道机理状态显著不同的候选合金。

---

## 主要脚本

### `build_mechanism_embeddings.py`

构建六个通道对应的参考机理空间。

主要操作包括：

- 仅使用机理文本进行 SciBERT 语义编码；
- 对最终层 token 表征进行均值池化；
- L2 归一化；
- 仅在训练集上拟合 PCA；
- 验证集/测试集仅使用训练集拟合后的 PCA 进行转换；
- 输出 64 维通道特异性参考机理向量；
- 输出 PCA 累积解释方差诊断结果。

canonical mechanism tags 仅用于机理空间的标注与解释，不参与语义嵌入输入。

### `pretrain_mechanism_encoder.py`

训练六通道机理监督编码器。

主要特征包括：

- 共享结构化输入编码器；
- 六个独立的 64 维机理输出头；
- 对 `not_reported` 状态采用通道特异性 mask；
- 使用 cosine-distance 监督；
- 固定的训练/验证/测试划分；
- 输入特征归一化仅在训练集上拟合；
- 依据验证集进行 checkpoint 选择；
- held-out 测试集不参与模型选择。

### `evaluate_pretraining.py`

在 held-out 测试集上评估最终预训练机理编码器。

该脚本从三个互补层面评估机理表征恢复能力：

- 预测机理向量与参考机理向量之间的逐点对齐；
- 各机理空间内部的成对相似关系保持；
- 对应 held-out 参考机理状态的检索能力。

报告指标包括 cosine-similarity 统计量、Spearman 相关系数、mean reciprocal rank、median reference rank、Recall@3 和 Recall@5。

该脚本对应 Fig. 3、Fig. 4、Table 1 和 Table 2。

### `train_frozen_transfer.py`

将预训练机理编码器冻结迁移至氧化动力学预测。

迁移过程中：

- 共享预训练编码器冻结；
- 六个机理头全部冻结；
- 仅优化动力学预测模块；
- 动力学预测头接收冻结的潜在材料表征；
- 六个机理向量继续作为并行模型输出；
- 使用 `lg(kp)` 的均方误差作为训练目标；
- 使用验证集 RMSE 进行 checkpoint 选择；
- 测试集不参与训练和模型选择。

### `evaluate_kinetics_release.py`

对冻结迁移模型进行最终 held-out 动力学评估。

报告：

- 决定系数（`R²`）；
- 平均绝对误差（`MAE`）；
- 均方根误差（`RMSE`）。

同时导出 held-out 动力学预测结果以及六通道预测机理向量，供后续机理解释使用。

该脚本对应 Fig. 5 及后续的机理 case 分析。

### `analyze_mechanism_cases.py`

对选定的 held-out 测试样本进行机理空间解释。

对于每个选定样本和每个机理通道：

1. 以预测得到的 64 维机理向量作为 query；
2. 从**训练集参考机理库**中检索 Top-5 最近邻机理状态；
3. 单独计算预测机理向量与 held-out 原始机理向量之间的 cosine similarity，用于后验验证；
4. 将预测状态、held-out 原始状态、Top-5 邻域以及训练集参考机理库投影到固定的通道特异性 UMAP 映射中。

held-out 原始机理不会参与 retrieval-grounded interpretation。

该脚本对应 Fig. 6、Fig. S9、Table S3 和 Table S4。

### `screen_mechanism_divergence_release.py`

对 Co 基候选合金状态进行前瞻性机理感知筛选。

搜索空间包括：

- Al：7–15 at.%；
- Cr：3–15 at.%；
- Ni：20–35 at.%；
- W：0–8 at.%；
- Ti：0–8 at.%；
- Mo：0–4 at.%；
- Ta：0–6 at.%；
- Si：0–3 at.%；

Co 作为平衡元素。

筛选流程包括 Latin hypercube sampling、0.5 at.% 离散化、可行性与重复组成筛除、冻结迁移模型对 `lg(kp)` 和六个机理表征的预测、成对动力学预筛选，以及基于训练参考机理库 cross-source 距离分布对各通道 cosine distance 进行归一化。

有效贡献通道数定义为

\[N_{\mathrm{eff}}=\frac{\left(\sum_c q_c\right)^2}{\sum_c q_c^2},\]

平衡多通道机理差异得分定义为

\[D_{\mathrm{mech}}=\bar{q}\sqrt{\frac{N_{\mathrm{eff}}}{6}}.\]

最终目标区域定义为

\[|\Delta \mathrm{lg}(k_p)| \leq 0.05\]

且

\[D_{\mathrm{mech}} \geq 0.60.\]

公开版本中不使用其他隐藏的 pair-selection criteria。

该脚本对应 Fig. 7、Fig. S10 和 Table S5。

---

## 超参数优化

`optimization/` 文件夹包含使用验证集进行模型超参数选择的 Optuna 工作流。

### `optimization/optimize_pretraining.py`

搜索参数包括：

- 网络深度；
- 隐藏层宽度；
- dropout；
- learning rate；
- weight decay。

Optuna 优化目标为**六个机理通道的平均验证集 cosine similarity**。held-out 测试集不参与 trial 排序、pruning、checkpoint 选择或超参数选择。

### `optimization/optimize_frozen_transfer.py`

搜索动力学预测头的：

- 网络深度；
- 隐藏层宽度；
- dropout；
- learning rate；
- weight decay。

在每个 trial 中，预训练编码器和六个机理头始终保持冻结。Optuna 优化目标为**验证集 RMSE**，held-out 测试集不参与超参数优化。

---

## Prompt 模板

`prompt/` 文件夹包含机理抽取与机理解释过程中使用的 LLM Prompt。

### `extraction_channel1-6_prompt.json`

定义联合六通道机理抽取框架，包括：

- 六个机理通道定义；
- 允许使用的 canonical tags；
- evidence-status 定义；
- 样本-条件对齐规则；
- 通道之间的边界；
- 跨通道一致性检查；
- 禁止的无证据推断；
- 规定的结构化 JSON 输出格式。

由该 Prompt 抽取得到的机理文本构成参考机理表征的文本基础。

### `retrieval_interpretation_prompts.json`

包含下游机理解释所使用的 Prompt，包括：

- 基于 Top-5 训练参考邻域生成通道级机理解释；
- 将六个通道级解释整合为总体氧化状态描述；
- 对预测机理与 held-out 文献机理进行后验比较；
- 分配 `Consistent`、`Partially consistent` 或 `Inconsistent` 一致性标签；
- 对代表性前瞻候选对进行 retrieval-grounded 机理比较。

对于 held-out case 分析，解释与验证严格分离：生成 retrieval-grounded interpretation 时不会提供 held-out 原始机理描述。

---

## Source data

`source_data/` 文件夹包含支撑本文公开图表的已处理数值与表格数据。

### 正文源数据

```text
source_data/
├── Fig.2/
├── Fig.3/
├── Fig.4/
├── Fig.5/
├── Fig.6/
├── Fig.7/
├── Table1-2/
└── Table4/
```

### 补充材料源数据

```text
source_data/SM/
├── Fig.S2-Fig.S7/
├── Fig.S8/
├── Fig.S9_TableS4/
├── Fig.S10_TableS5/
├── TableS2/
└── TableS3/
```

这些 source-data 文件夹包含用于复现或核查相应图表的已处理数据。

---

## 推荐运行顺序

```text
1. build_mechanism_embeddings.py
        ↓
2. optimization/optimize_pretraining.py
        ↓
3. pretrain_mechanism_encoder.py
        ↓
4. evaluate_pretraining.py
        ↓
5. optimization/optimize_frozen_transfer.py
        ↓
6. train_frozen_transfer.py
        ↓
7. evaluate_kinetics_release.py
        ↓
8. analyze_mechanism_cases.py
        ↓
9. screen_mechanism_divergence_release.py
```

---

## 软件环境

构建文献语义参考机理表征时使用的 SciBERT 模型为：

```text
allenai/scibert_scivocab_uncased
```

---

## 数据可用性

用于支撑公开图表及主要定量分析的已处理数据均提供于 `source_data/`。

用于模型开发的完整 literature-derived dataset，包括完整的样本-条件级结构化数据集以及完整机理语料，**未包含在本仓库中，因为其仍属于正在进行的研究工作的一部分**。

因此，本仓库支持对已公开图表源数据的检查与复现，并完整记录论文中的建模工作流；但如果需要从头完成端到端模型重新训练，则仍需获得底层 literature-derived development dataset。

---
