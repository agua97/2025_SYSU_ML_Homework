import pandas as pd
import pickle
import os
import numpy as np
import random

# ==================== 📂 路径配置 ====================
# 1. 原始数据目录
BASE_DATA_DIR = r'../Agent4Rec-master/datasets/ml-1m/cf_data'
# 2. Agent 仿真日志目录
PKL_FOLDER_PATH = r'..\Agent4Rec-master\storage\ml-1m\LightGCN\lgn_1000_5_4_1009\behavior_clean'
# 3. 输出目录
RECHORUS_DATA_ROOT = '../data'
# ====================================================

def read_cf_txt(filepath):
    print(f"📖 读取原始文件: {filepath}")
    data = []
    if not os.path.exists(filepath): return pd.DataFrame(), {}
    user_history = {} 
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            user_id = int(parts[0])
            items = [int(i) for i in parts[1:]]
            if user_id not in user_history: user_history[user_id] = set()
            user_history[user_id].update(items)
            for item in items: data.append([user_id, item])
    return pd.DataFrame(data, columns=['user_id', 'item_id']), user_history

def parse_agent_data_all(folder_path):
    """
    【Variant A 核心逻辑】
    提取所有 watch_id，不管评分是多少，全部保留。
    """
    print(f"🤖 解析 Agent 数据 (全量无清洗模式)...")
    new_interactions = []
    if not os.path.exists(folder_path): return pd.DataFrame()

    files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    for filename in files:
        try:
            user_id = int(filename.split('.')[0])
            with open(os.path.join(folder_path, filename), 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                for page, content in data.items():
                    # 直接获取看过的列表
                    watch_ids = content.get('watch_id', [])
                    
                    if isinstance(watch_ids, (np.ndarray, list)):
                        for i in range(len(watch_ids)):
                            # 只要看了就加进去，不判断 rating
                            new_interactions.append([user_id, int(watch_ids[i])])
        except: continue
    return pd.DataFrame(new_interactions, columns=['user_id', 'item_id'])

def generate_negative_samples(df_target, global_history, all_items, num_neg=99):
    print(f"🎲 生成负样本...")
    neg_lists = []
    all_items_list = list(all_items)
    for idx, row in df_target.iterrows():
        u, i = row['user_id'], row['item_id']
        seen = global_history.get(u, set())
        seen.add(i)
        samples = []
        while len(samples) < num_neg:
            candidates = random.sample(all_items_list, min(len(all_items_list), num_neg * 2))
            for cand in candidates:
                if cand not in seen and cand not in samples:
                    samples.append(cand)
                    if len(samples) == num_neg: break
        neg_lists.append(str(samples))
    return neg_lists

def save_dataset(df_train, df_valid, df_test, folder_name):
    target_dir = os.path.join(RECHORUS_DATA_ROOT, folder_name)
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    print(f"💾 保存数据集至: {target_dir}")
    
    df_train[['user_id', 'item_id', 'time']].to_csv(
        os.path.join(target_dir, 'train.csv'), sep='\t', index=False, header=['user_id', 'item_id', 'time'])
    
    df_valid[['user_id', 'item_id', 'time', 'neg_items']].to_csv(
        os.path.join(target_dir, 'dev.csv'), sep='\t', index=False, header=['user_id', 'item_id', 'time', 'neg_items'])
    
    df_test[['user_id', 'item_id', 'time', 'neg_items']].to_csv(
        os.path.join(target_dir, 'test.csv'), sep='\t', index=False, header=['user_id', 'item_id', 'time', 'neg_items'])

def main():
    # 1. 读取原始数据
    df_train_orig, hist_train = read_cf_txt(os.path.join(BASE_DATA_DIR, 'train.txt'))
    df_valid_orig, hist_valid = read_cf_txt(os.path.join(BASE_DATA_DIR, 'valid.txt'))
    df_test_orig, hist_test  = read_cf_txt(os.path.join(BASE_DATA_DIR, 'test.txt'))
    
    # 补充时间戳 (ReChorus必需)
    df_train_orig['time'] = 1
    df_valid_orig['time'] = 2
    df_test_orig['time'] = 3

    # 2. 读取 Agent 全量数据 (无清洗)
    df_agent_all = parse_agent_data_all(PKL_FOLDER_PATH)
    df_agent_all['time'] = 1
    
    print(f"原始训练集数量: {len(df_train_orig)}")
    print(f"Agent新增数量 (Variant A): {len(df_agent_all)}")

    # 3. 合并训练集 (Origin + Agent All)
    # 去重：防止 Agent 看了用户本来就看过的电影
    df_train_variant_a = pd.concat([df_train_orig, df_agent_all], ignore_index=True)
    df_train_variant_a.drop_duplicates(subset=['user_id', 'item_id'], inplace=True)

    # 4. 生成负样本 (Dev/Test)
    # 注意：负样本必须避开 (训练集 + Agent全量数据 + 验证测试集)
    all_items = set()
    all_items.update(df_train_variant_a['item_id'].unique())
    all_items.update(df_valid_orig['item_id'].unique())
    all_items.update(df_test_orig['item_id'].unique())
    
    global_hist = {}
    def merge_h(h):
        for u, items in h.items():
            if u not in global_hist: global_hist[u] = set()
            global_hist[u].update(items)
    merge_h(hist_train); merge_h(hist_valid); merge_h(hist_test)
    
    # 把 Agent 的数据也加入历史，防止负样本采到
    for idx, row in df_agent_all.iterrows():
        u, i = row['user_id'], row['item_id']
        if u not in global_hist: global_hist[u] = set()
        global_hist[u].add(i)

    # 生成负样本
    df_valid_orig['neg_items'] = generate_negative_samples(df_valid_orig, global_hist, all_items)
    df_test_orig['neg_items'] = generate_negative_samples(df_test_orig, global_hist, all_items)

    # 5. 保存
    save_dataset(df_train_variant_a, df_valid_orig, df_test_orig, 'Agent4Rec_All')
    
    print("✅ 完成！请使用数据集 'Agent4Rec_All' 运行消融实验。")

if __name__ == '__main__':
    main()