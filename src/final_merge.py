import pandas as pd
import pickle
import os
import numpy as np
import random
import ast

# ==================== 📂 路径配置 ====================
# 1. Agent4Rec 的基准数据目录
BASE_DATA_DIR = r'../Agent4Rec-master/datasets/ml-1m/cf_data'

# 2. Agent 仿真数据的 pkl 文件夹
PKL_FOLDER_PATH = r'..\Agent4Rec-master\storage\ml-1m\LightGCN\lgn_1000_5_4_1009\behavior_clean'

# 3. ReChorus 的数据根目录
RECHORUS_DATA_ROOT = '../data'
# ====================================================

def read_cf_txt(filepath):
    """读取 txt 邻接表文件"""
    print(f"📖 读取文件: {filepath}")
    data = []
    if not os.path.exists(filepath):
        print(f"❌ 错误: 找不到文件 {filepath}")
        return pd.DataFrame(), {}
    
    # 用字典记录每个用户的交互历史，用于负采样去重
    user_history = {} 
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            user_id = int(parts[0])
            items = [int(i) for i in parts[1:]]
            
            # 记录历史
            if user_id not in user_history:
                user_history[user_id] = set()
            user_history[user_id].update(items)
            
            for item in items:
                data.append([user_id, item])
                
    df = pd.DataFrame(data, columns=['user_id', 'item_id'])
    return df, user_history

def parse_agent_viewed_data(folder_path):
    """解析 Agent 的 pkl 日志"""
    print(f"🤖 解析 Agent 仿真数据: {folder_path}")
    new_interactions = []
    
    if not os.path.exists(folder_path):
        return pd.DataFrame()

    files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    
    for filename in files:
        try:
            user_id = int(filename.split('.')[0])
            with open(os.path.join(folder_path, filename), 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                for page, content in data.items():
                    watch_ids = content.get('watch_id', [])
                    ratings = content.get('rating', [])
                    if isinstance(watch_ids, (np.ndarray, list)):
                        for i in range(len(watch_ids)):
                         # 只有当评分存在且 >= 4 时才加入
                            if i < len(ratings) and int(ratings[i]) >= 4:
                                new_interactions.append([user_id, int(watch_ids[i])])
        except Exception:
            continue
            
    df = pd.DataFrame(new_interactions, columns=['user_id', 'item_id'])
    return df

def generate_negative_samples(df_target, global_history, all_items, num_neg=99):
    """
    核心函数：为测试集/验证集的每一行生成 99 个负样本
    """
    print(f"🎲 正在为 {len(df_target)} 条数据生成 99 个负样本 (这可能需要几秒钟)...")
    
    neg_lists = []
    all_items_list = list(all_items) # 转成列表以便采样
    
    for idx, row in df_target.iterrows():
        u = row['user_id']
        i = row['item_id']
        
        # 该用户看过的所有电影 (Train + Valid + Test + AgentHistory)
        seen = global_history.get(u, set())
        # 还要加上当前这一条测试数据的 item (防止漏掉)
        seen.add(i)
        
        samples = []
        while len(samples) < num_neg:
            candidates = random.sample(all_items_list, min(len(all_items_list), num_neg * 2))
            for cand in candidates:
                if cand not in seen and cand not in samples:
                    samples.append(cand)
                    if len(samples) == num_neg:
                        break
        
        # 转成字符串格式 "[1, 2, 3]" 方便 csv 保存
        neg_lists.append(str(samples))
        
    return neg_lists

def save_to_folder(df_train, df_valid, df_test, folder_name):
    target_dir = os.path.join(RECHORUS_DATA_ROOT, folder_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    print(f"💾 保存数据集到: {target_dir}")
    
    # 保存 Train (不需要负样本)
    df_train = df_train[['user_id', 'item_id', 'time']]
    df_train.to_csv(os.path.join(target_dir, 'train.csv'), sep='\t', index=False, header=['user_id', 'item_id', 'time'])
    
    # 保存 Dev (带负样本)
    df_valid = df_valid[['user_id', 'item_id', 'time', 'neg_items']]
    df_valid.to_csv(os.path.join(target_dir, 'dev.csv'), sep='\t', index=False, header=['user_id', 'item_id', 'time', 'neg_items'])
    
    # 保存 Test (带负样本)
    df_test = df_test[['user_id', 'item_id', 'time', 'neg_items']]
    df_test.to_csv(os.path.join(target_dir, 'test.csv'), sep='\t', index=False, header=['user_id', 'item_id', 'time', 'neg_items'])
    
    print(f"   Train: {len(df_train)} | Dev: {len(df_valid)} | Test: {len(df_test)}")

def main():
    # 1. 读取所有原始数据
    print("⏳ [Step 1] 读取原始数据...")
    df_train_orig, hist_train = read_cf_txt(os.path.join(BASE_DATA_DIR, 'train.txt'))
    df_valid_orig, hist_valid = read_cf_txt(os.path.join(BASE_DATA_DIR, 'valid.txt'))
    df_test_orig, hist_test  = read_cf_txt(os.path.join(BASE_DATA_DIR, 'test.txt'))
    
    # 2. 读取 Agent 数据
    print("\n⏳ [Step 2] 读取 Agent 数据...")
    df_agent = parse_agent_viewed_data(PKL_FOLDER_PATH)
    
    # =============== 🔴 新增：严格过滤逻辑 (复现论文) ===============
    print("\n✂️ [Filter] 正在过滤非 Simulation 用户...")
    
    # 1. 获取那 1000 个模拟用户的 ID 列表
    target_users = df_agent['user_id'].unique()
    target_uid_set = set(target_users)
    print(f"   检测到仿真用户数: {len(target_uid_set)}")

    # 2. 过滤原始数据集，只保留这 1000 人
    # 注意：这里我们覆盖原变量，把不相关的人剔除
    df_train_orig = df_train_orig[df_train_orig['user_id'].isin(target_uid_set)].copy()
    df_valid_orig = df_valid_orig[df_valid_orig['user_id'].isin(target_uid_set)].copy()
    df_test_orig  = df_test_orig[df_test_orig['user_id'].isin(target_uid_set)].copy()

    # 3. 必须同步过滤 hist (否则负采样会出错)
    # 重新构建只包含这 1000 人的历史字典
    def filter_hist(old_hist):
        return {u: items for u, items in old_hist.items() if u in target_uid_set}

    hist_train = filter_hist(hist_train)
    hist_valid = filter_hist(hist_valid)
    hist_test = filter_hist(hist_test)

    print(f"   过滤后剩余记录数 -> Train: {len(df_train_orig)} | Valid: {len(df_valid_orig)} | Test: {len(df_test_orig)}")
    # ==============================================================
    
    
    # 3. 构建全局 Item 集合 和 全局 User History (用于负采样去重)
    print("\n🏗️ [Step 3] 构建全局索引...")
    # 获取所有出现过的 Item ID
    all_items = set()
    all_items.update(df_train_orig['item_id'].unique())
    all_items.update(df_valid_orig['item_id'].unique())
    all_items.update(df_test_orig['item_id'].unique())
    all_items.update(df_agent['item_id'].unique()) # Agent 新发现的物品也算
    
    print(f"   系统中共有 {len(all_items)} 个不同的 Item。")
    
    # 合并用户历史：History = Train + Valid + Test + AgentViewed
    # 这样在采样负样本时，绝对不会采到用户以前看过的
    global_history = {}
    
    # 辅助函数：合并 history
    def merge_hist(source_hist):
        for u, items in source_hist.items():
            if u not in global_history: global_history[u] = set()
            global_history[u].update(items)
    
    merge_hist(hist_train)
    merge_hist(hist_valid)
    merge_hist(hist_test)
    
    # 把 Agent 的数据也加进历史 (针对 Enhanced 组，但为了方便，Baseline 组也可以共用这个排除逻辑，因为反正本来就没看过)
    for idx, row in df_agent.iterrows():
        u = row['user_id']
        i = row['item_id']
        if u not in global_history: global_history[u] = set()
        global_history[u].add(i)

    # 4. 为 Dev 和 Test 生成负样本 (计算量较大，做一次即可)
    print("\n🎲 [Step 4] 生成负样本 (99个/条)...")
    neg_valid = generate_negative_samples(df_valid_orig, global_history, all_items)
    neg_test = generate_negative_samples(df_test_orig, global_history, all_items)
    
    # 将负样本挂载到 DataFrame
    df_valid_orig['neg_items'] = neg_valid
    df_test_orig['neg_items'] = neg_test
    
    # 补充 time 列
    df_train_orig['time'] = 1
    df_valid_orig['time'] = 2
    df_test_orig['time'] = 3
    df_agent['time'] = 1

    # 5. 保存 Baseline
    print("\n📦 [Step 5] 保存 Baseline 数据集...")
    save_to_folder(df_train_orig, df_valid_orig, df_test_orig, 'AgentRec_Original')

    # 6. 保存 Enhanced
    print("\n📦 [Step 6] 保存 Enhanced 数据集...")
    # 合并 Agent 数据到训练集
    df_train_enhanced = pd.concat([df_train_orig, df_agent], ignore_index=True)
    df_train_enhanced.drop_duplicates(subset=['user_id', 'item_id'], inplace=True)
    
    # Dev 和 Test 保持不变 (包含刚才生成的 neg_items)
    save_to_folder(df_train_enhanced, df_valid_orig, df_test_orig, 'AgentRec_Enhanced')

    print("\n🎉 完美解决！现在 dev.csv 和 test.csv 里面包含了真实的 99 个负样本列表。")
    print("格式示例: \"[120, 45, 999, ...]\"")

if __name__ == '__main__':
    main()