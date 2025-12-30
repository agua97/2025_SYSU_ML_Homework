import pandas as pd
import os

# ===================== ⚙️ 配置区域 =====================
BASE_SUBSET_DIR = "../data/Grocery_Subset"      # 原始干净子集
ENHANCED_DIR = "../data/Grocery_Agent_Enhanced" # 增强后的数据集存放处
FEEDBACK_LOG = "../data/Grocery_Subset/agent_feedback.csv"             # 刚才仿真的输出

def main():
    if not os.path.exists(ENHANCED_DIR): os.makedirs(ENHANCED_DIR)

    print("📖 正在加载数据...")
    # 1. 加载原始人类行为
    df_train_orig = pd.read_csv(os.path.join(BASE_SUBSET_DIR, "train.csv"), sep="\t")
    df_dev = pd.read_csv(os.path.join(BASE_SUBSET_DIR, "dev.csv"), sep="\t")
    test = pd.read_csv(os.path.join(BASE_SUBSET_DIR, "test.csv"), sep="\t")

    # 2. 加载 Agent 仿真行为
    df_agent = pd.read_csv(FEEDBACK_LOG, sep="\t")

    # 3. 执行论文逻辑：筛选高分行为 (Rating >= 4)
    # 论文认为，只有 Agent 给出高分的推荐，才代表模型学到了有用的偏好
    print(f"🤖 原始 Agent 日志数: {len(df_agent)}")
    df_agent_hq = df_agent[df_agent['rating'] >= 4].copy()
    print(f"✨ 筛选高分反馈 (Rating >= 4) 后剩余: {len(df_agent_hq)}")

    # 4. 构造增强训练集 (Human + Agent)
    # 只取 ReChorus 需要的三列：user_id, item_id, time
    df_train_enhanced = pd.concat([
        df_train_orig[['user_id', 'item_id', 'time']], 
        df_agent_hq[['user_id', 'item_id', 'time']]
    ], ignore_index=True)

    # 去重（防止 Agent 买了人类已经买过的东西）
    df_train_enhanced.drop_duplicates(subset=['user_id', 'item_id'], inplace=True)

    # 5. 按照 ReChorus 标准格式保存
    print(f"💾 正在保存增强数据集至: {ENHANCED_DIR}")
    df_train_enhanced.to_csv(os.path.join(ENHANCED_DIR, "train.csv"), sep="\t", index=False)
    
    df_dev.to_csv(os.path.join(ENHANCED_DIR, "dev.csv"), sep="\t", index=False)
    test.to_csv(os.path.join(ENHANCED_DIR, "test.csv"), sep="\t", index=False)

    print("\n🎉 任务完成！")
    print(f"原始训练集大小: {len(df_train_orig)}")
    print(f"增强后训练集大小: {len(df_train_enhanced)}")
    print(f"现在你可以运行: python main.py --model_name LightGCN --dataset Grocery_Agent_Enhanced")

if __name__ == "__main__":
    main()