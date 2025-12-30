import pandas as pd
import numpy as np
import os
import json
from openai import OpenAI
from tqdm import tqdm

# ===================== ⚙️ 配置区域 =====================
DATA_DIR = "../data/Grocery_Subset"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
META_PATH = os.path.join(DATA_DIR, "item_meta_enriched.csv")
OUTPUT_PATH = "agent_profiles.json"

# API 配置 (用于生成自然语言 Persona)
client = OpenAI(api_key="your_api_key", base_url="https://api.siliconflow.cn/v1/")
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2" # 或 gpt-3.5-turbo

# ===================== 🧮 第一部分：计算社会特质 =====================

def calculate_social_traits():
    print("📊 正在计算 Agent 社会特质 (Activity, Diversity, Conformity)...")
    train = pd.read_csv(TRAIN_PATH, sep="\t")
    meta = pd.read_csv(META_PATH, sep="\t")
    
    # 合并元数据
    df = train.merge(meta, on='item_id', how='left')
    
    # 1. Activity (活动度): 
    user_activity = df.groupby('user_id').size().rename('activity_score')
    
    # 2. Diversity (多样性): 
    user_diversity = df.groupby('user_id')['i_category'].nunique().rename('diversity_score')
    
    # 3. Conformity (从众度): 
    item_pop = train.groupby('item_id').size().rename('item_popularity')
    df = df.merge(item_pop, on='item_id', how='left')
    
    if 'rating' in df.columns:
        # 论文公式: 用户评分与全局平均分的均方误差
        item_avg_rating = df.groupby('item_id')['rating'].mean().rename('avg_r')
        df = df.merge(item_avg_rating, on='item_id')
        df['diff'] = (df['rating'] - df['avg_r'])**2
        user_conformity = 1 / df.groupby('user_id')['diff'].mean().rename('conformity_score') # 取倒数，偏差越小从众度越高
    else:
        # 隐性反馈逻辑: 购买物品的平均流行度
        user_conformity = df.groupby('user_id')['item_popularity'].mean().rename('conformity_score')

    # 汇总特质
    traits = pd.concat([user_activity, user_diversity, user_conformity], axis=1).fillna(0)
    
    # 按照论文进行三级离散化 (Low, Medium, High)
    def discretize(series):
        try:
            return pd.qcut(series, 3, labels=["Low", "Medium", "High"], duplicates='drop')
        except:
            return pd.Series(["Medium"] * len(series), index=series.index)

    traits['activity_level'] = discretize(traits['activity_score'])
    traits['diversity_level'] = discretize(traits['diversity_score'])
    traits['conformity_level'] = discretize(traits['conformity_score'])
    
    return traits, df

# ===================== 🤖 第二部分：生成自然语言 Persona =====================

def generate_persona(uid, user_history_df):
    """
    对齐论文 2.1.1: 选取历史记录，让 LLM 总结用户独特口味
    """
    # 选取最近的 15 条历史 (论文建议采样，Grocery 数据通常较少，取 15 条即可)
    history = user_history_df.tail(15)
    items_str = []
    for _, row in history.iterrows():
        items_str.append(f"- {row['title']} ({row['i_category']})")
    
    history_context = "\n".join(items_str)
    
    prompt = f"""
    基于以下用户的生鲜购物历史，请用两句话总结该用户的购物习惯和潜在偏好。
    不要出现姓名，直接描述特征。
    
    用户购买历史：
    {history_context}
    
    总结样例：该用户关注健康生活，偏好有机蔬菜和低脂乳制品。在购物时表现出较高的品牌忠诚度，倾向于回购常用的烹饪调料。
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "该用户是一位常规的生鲜购物者，偏好基础食品和日常消耗品。"

# ===================== 🚀 主程序逻辑 =====================

def main():
    # 1. 计算量化特质
    traits, full_df = calculate_social_traits()
    
    # 2. 遍历 1000 个 Agent 生成完整画像
    final_profiles = {}
    target_users = traits.index.tolist()
    
    print(f"🤖 正在调用 LLM 为 {len(target_users)} 个 Agent 生成个性化人设 (Persona)...")
    
    for uid in tqdm(target_users):
        user_history = full_df[full_df['user_id'] == uid]
        persona = generate_persona(uid, user_history)
        
        final_profiles[int(uid)] = {
            "traits": {
                "activity": traits.loc[uid, 'activity_level'],
                "diversity": traits.loc[uid, 'diversity_level'],
                "conformity": traits.loc[uid, 'conformity_level']
            },
            "persona": persona,
            "raw_scores": {
                "activity": float(traits.loc[uid, 'activity_score']),
                "diversity": float(traits.loc[uid, 'diversity_score']),
                "conformity": float(traits.loc[uid, 'conformity_score'])
            }
        }
    
    # 3. 保存结果
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_profiles, f, ensure_ascii=False, indent=4)
    
    print(f"✨ 成功！Agent 画像已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()