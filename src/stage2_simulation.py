# -*- coding: utf-8 -*-
import os
import sys
import io
import json
import re
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# 环境加固
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ===================== ⚙️ 配置区域 =====================
MY_API_KEY = "your_api_key"
MY_BASE_URL = "https://api.siliconflow.cn/v1/" 
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

PROFILES_PATH = "agent_profiles.json"
CANDIDATES_PATH = "../log/LightGCN/LightGCN__Grocery_Subset__0__lr=0/rec-LightGCN-test.csv" 
META_PATH = "../data/Grocery_Subset/item_meta_enriched.csv"
FEEDBACK_LOG_PATH = "../data/Grocery_Subset/agent_feedback.csv"

ITEMS_PER_PAGE = 5  
MAX_PAGES = 4       

client = OpenAI(api_key=MY_API_KEY, base_url=MY_BASE_URL)

# ===================== 🛠️ 辅助加载函数 =====================

def load_candidates():
    df = pd.read_csv(CANDIDATES_PATH, sep="\t")
    def parse_ids(s):
        return [int(x) for x in re.findall(r'\d+', str(s))]
    return df.set_index('user_id')['rec_items'].apply(parse_ids).to_dict()

def load_meta():
    meta = pd.read_csv(META_PATH, sep="\t", encoding='utf-8')
    return meta.set_index('item_id')[['title', 'i_category']].to_dict('index')

# ===================== 🤖 核心仿真逻辑 =====================

def simulate_agent_session(uid, profile, candidates_dict, meta_map):
    session_history = []
    # Unicode 转义处理中文
    raw_persona = profile.get('persona', '')
    safe_persona = raw_persona.encode('unicode_escape').decode('ascii')
    
    traits = profile.get('traits', {})
    user_rec_pool = candidates_dict.get(uid, [])
    if not user_rec_pool: return []

    pages_visited = 0
    items_bought = 0

    for page_idx in range(MAX_PAGES):
        pages_visited += 1
        start = page_idx * ITEMS_PER_PAGE
        end = start + ITEMS_PER_PAGE
        page_ids = user_rec_pool[start:end]
        if not page_ids: break
        
        page_items_list = []
        for iid in page_ids:
            info = meta_map.get(iid, {"title": "Unknown Item", "i_category": "Grocery"})
            page_items_list.append(f"ID:{iid} | Title:{info['title']} | Category:{info['i_category']}")
        
        items_block = "\n".join(page_items_list)

        prompt = f"""
### USER BACKGROUND
- Personal Identity (Unicode): {safe_persona}
- Shopping Traits: Activity:{traits.get('activity')}, Diversity:{traits.get('diversity')}, Conformity:{traits.get('conformity')}

### CONTEXT
You are browsing a grocery store mobile app. Page: {page_idx + 1}.

### RECOMMENDED ITEMS
{items_block}

### YOUR TASK
Based on your identity, provide:
1. BUY: IDs and Rating (1-5).
2. EXIT: Decide if you want to continue browsing.

Respond strictly in JSON:
{{
    "purchases": [{{"id": 101, "rating": 5}}],
    "continue_to_next_page": true/false
}}
"""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" },
                timeout=60, # 增加超时保护
                temperature=0.3
            )
            res = json.loads(response.choices[0].message.content)
            
            current_buys = res.get('purchases', [])
            for action in current_buys:
                session_history.append({"user_id": uid, "item_id": action['id'], "rating": action['rating']})
                items_bought += 1
            
            if not res.get('continue_to_next_page', True): 
                break
        except Exception as e:
            # 使用 tqdm.write 避免破坏进度条
            tqdm.write(f"  [Error] User {uid} at Page {page_idx+1}: {str(e)}")
            break
            
    return session_history, pages_visited, items_bought

# ===================== 🚀 主程序 =====================

def main():
    tqdm.write("Step 1: Loading data files...")
    with open(PROFILES_PATH, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    candidates_dict = load_candidates()
    meta_map = load_meta()
    
    all_agent_logs = []
    target_uids = list(profiles.keys())
    
    total_purchases = 0
    
    tqdm.write(f"Step 2: Starting simulation for {len(target_uids)} Agents...")
    
    # 动态进度条
    pbar = tqdm(target_uids, ascii=True, desc="Simulating")
    
    for i, uid_str in enumerate(pbar):
        uid = int(uid_str)
        session_data, p_count, b_count = simulate_agent_session(uid, profiles[uid_str], candidates_dict, meta_map)
        
        all_agent_logs.extend(session_data)
        total_purchases += b_count
        
        # 找到循环中这一段，进行修改：
        if (i + 1) % 2 == 0:  # 改成每 2 个用户打一次，更频繁地看到反馈
            t_traits = profiles[uid_str]['traits']
            # 确保 persona 的中文在打印时不会出问题，手动处理一下编码
            try:
                t_persona = profiles[uid_str]['persona'][:30]
                # 使用标准的 print 配合 flush=True 强制刷新到屏幕
                print(f"\n[Log] User {uid} ({t_traits['activity']}): Viewed {p_count} pages, Bought {b_count} items. Persona: {t_persona}", flush=True)
            except:
                print(f"\n[Log] User {uid}: Viewed {p_count} pages, Bought {b_count} items.", flush=True)
        
        # 更新进度条右侧
        pbar.set_postfix({"Total_Buys": total_purchases})
    
    if all_agent_logs:
        df = pd.DataFrame(all_agent_logs)
        
        # --- 新增逻辑：对齐时间戳 ---
        # 读取原始训练集的最大时间，让 Agent 的行为发生在“未来”
        try:
            orig_train = pd.read_csv("../data/Grocery_Subset/train.csv", sep="\t")
            max_time = orig_train['time'].max()
        except:
            max_time = 1000000 # 如果找不到，给个基准值
            
        # 给 Agent 行为分配时间戳（简单起见，全部设为 max_time + 1）
        df['time'] = max_time + 1
        
        # 按照 ReChorus 的习惯，用 Tab 分隔保存
        df.to_csv(FEEDBACK_LOG_PATH, index=False, sep="\t", encoding='utf_8_sig')
        print(f"Success! Saved {len(df)} logs to {FEEDBACK_LOG_PATH} (Tab-separated with time)")
        tqdm.write(f"Step 3: Success! Total {len(df)} logs saved to {FEEDBACK_LOG_PATH}")
    else:
        tqdm.write("No data generated.")

if __name__ == "__main__":
    main()