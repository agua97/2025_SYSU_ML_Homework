# Agent4Rec + ReChorus 增强推荐模型复现指南

## 📝 项目结构说明
本项目旨在利用生成式智能体对推荐系统进行数据增强。所有核心脚本均存放于 `src/` 目录下。

* **src/**：核心代码库。包含仿真、数据清洗、ReChorus 适配脚本及训练入口。
* **Agent4Rec-master/**：MovieLens 数据中心。直接使用该目录下的原始数据及仿真生成的 `.pkl` 日志。
* **data/**：数据输出中心。存放下载的 Amazon 原始 JSON 及脚本运行生成的训练集。

---

## 📂 数据集准备 (如果只是想测试代码是否跑通，此处可跳过)

### 1. MovieLens-1M (ML-1M)
本项直接利用 `Agent4Rec-master` 仓库中的数据结构，请确保以下文件存在：
* **基础邻接表**：`Agent4Rec-master/datasets/ml-1m/cf_data/` 目录下需包含 `train.txt`, `valid.txt`, `test.txt`。
* **仿真日志**：`Agent4Rec-master/storage/ml-1m/LightGCN/lgn_1000_5_4_1009/behavior_clean/` 目录下需存有 1000 个智能体的 `.pkl` 行为日志。

### 2. Amazon Grocery and Gourmet Food
本实验需要手动下载原始 Amazon 数据集并进行文本恢复。

#### **📥 官方数据下载地址 (2018 Edition)**
请从以下官方入口获取必要数据文件：
* **交互数据 (5-core)**: [reviews_Grocery_and_Gourmet_Food_5.json.gz](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Grocery_and_Gourmet_Food_5.json.gz)
* **元数据 (Metadata)**: [meta_Grocery_and_Gourmet_Food.json.gz](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Grocery_and_Gourmet_Food.json.gz)

> **放置路径**：下载后请**不要解压**，直接放入：`data/Grocery_and_Gourmet_Food/`

## 环境配置
实验需在 Python 3.9 环境中运行，核心依赖包括：
* Python 3.9
* PyTorch 2.5.1 + cu118
* OpenAI 2.14.0 (用于调用 DeepSeek-V3.2)
* Pandas 2.3.3
* NumPy 1.23.5

完整环境配置详见 `requirements.txt` 文件，可通过以下命令创建环境：

```bash
conda create -n rechorus python=3.9
conda activate rechorus
pip install -r requirements.txt
```

## 🚀 完整执行流水线

### 第一阶段：Amazon Grocery 全链路仿真
此阶段将完成从“数据下载”到“生成增强训练集”的全过程：

1. **文本恢复与 ID 映射**（解析原始 JSON）：
   ```bash
   python src/recover_text.py
   ```

2. **筛选 1,000 名活跃用户子集**：
   ```bash
   python src/stage_0_prepare_1000_agents.py
   ```

3. **画像生成与交互仿真（需配置 API Key）**：
   ```bash
   python src/stage1_profile.py
   python src/stage2_simulation.py
   ```

4. **适配 ReChorus 格式并合并数据**：
   ```bash
   python src/stage3_merge_for_rechorus.py
   ```

### 第二阶段：MovieLens 日志提取与精炼
利用本组编写的适配器直接从 Agent4Rec 日志中提炼交互：

1. **提取评分 ≥ 4 的 Enhanced 组并生成 1:99 负样本**：
   ```bash
   python src/final_merge.py
   ```

2. **提取全量模拟数据用于消融实验 (Variant A)**：
   ```bash
   python src/merge_all.py
   ```

## 📊 模型训练与评测

进入 `src` 目录，在增强后的数据集上重训并评测。

> **关键设置**：必须携带 `--test_all 0` 以启用固化的 1:99 负采样评测协议。

### **示例：在 Grocery 增强数据集上训练 LightGCN**
  ```bash
   python src/main.py --model_name LightGCN --dataset Grocery_Agent_Enhanced --path ./data/ --lr 1e-3 --l2 1e-4 --batch_size 2048 --epoch 100 --test_all 0 --regenerate 1
  ```

### **示例：在 Movielens 增强数据集上训练 LightGCN**
  ```bash
   python src/main.py --model_name LightGCN --dataset Agent4Rec_Enhanced --path ./data/ --lr 1e-3 --l2 1e-4 --batch_size 2048 --epoch 100 --test_all 0 --regenerate 1
  ```

## 📂 核心代码功能速查

| 文件名 | 功能描述 |
| :--- | :--- |
| **recover_text.py** | 自动检查原始 JSON 文件，并重建 Item ID 到标题/类别的映射。 |
| **final_merge.py** | MovieLens 适配器：解析日志、执行 Rating ≥ 4 过滤并固化负采样列表。 |
| **stage2_simulation.py** | LLM 仿真引擎：实现智能体 Page-by-page 浏览与动态退出逻辑。 |
| **stage3_merge_for_rechorus.py** | 数据合并工具：将原始人类行为与 Agent 补全行为合并，生成 CSV 训练集。 |
