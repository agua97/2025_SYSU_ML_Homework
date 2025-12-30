import pandas as pd
import os
import random

# 配置路径
ORIGINAL_DATA_DIR = "../data/Grocery_and_Gourmet_Food"
TARGET_DIR = "../data/Grocery_Subset" # 论文逻辑中的 Baseline 子集
MIN_INTERACTIONS = 15 # 建议设为 15，确保 Agent 有足够的 Context 供 LLM 学习

def prepare_data():
    if not os.path.exists(TARGET_DIR): os.makedirs(TARGET_DIR)

    # 1. 加载全量训练数据
    print("读取原始数据...")
    train = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, "train.csv"), sep="\t")
    dev = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, "dev.csv"), sep="\t")
    test = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, "test.csv"), sep="\t")

    # 2. 找出活跃用户池 (交互次数 >= 15)
    user_counts = train.groupby('user_id').size()
    active_user_pool = user_counts[user_counts >= MIN_INTERACTIONS].index.tolist()
    print(f"符合条件的用户总数: {len(active_user_pool)}")

    if len(active_user_pool) < 1000:
        print("警告：符合条件的用户不足 1000，将全部选中。")
        target_users = active_user_pool
    else:
        # 3. 随机抽取 1000 个用户 (固定种子保证可复现)
        random.seed(42)
        target_users = random.sample(active_user_pool, 1000)

    # 4. 根据选中 ID 过滤所有数据
    train_sub = train[train['user_id'].isin(target_users)]
    dev_sub = dev[dev['user_id'].isin(target_users)]
    test_sub = test[test['user_id'].isin(target_users)]

    # 5. 保存到新文件夹
    train_sub.to_csv(os.path.join(TARGET_DIR, "train.csv"), sep="\t", index=False)
    dev_sub.to_csv(os.path.join(TARGET_DIR, "dev.csv"), sep="\t", index=False)
    test_sub.to_csv(os.path.join(TARGET_DIR, "test.csv"), sep="\t", index=False)
    
    # 6. 同时把 item_meta 也复制过去，方便 ReChorus 读取
    meta = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, "item_meta_enriched.csv"), sep="\t")
    meta.to_csv(os.path.join(TARGET_DIR, "item_meta_enriched.csv"), sep="\t", index=False)

    # 保存用户 ID 列表，后续仿真要用
    with open("target_user_ids.txt", "w") as f:
        for u in target_users:
            f.write(f"{u}\n")

    print(f"✅ 子集准备就绪！存放在: {TARGET_DIR}")
    print(f"用户数: {len(target_users)}, 训练样本数: {len(train_sub)}")

if __name__ == "__main__":
    prepare_data()