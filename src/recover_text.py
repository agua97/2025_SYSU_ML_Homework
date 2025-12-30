import os
import gzip
import subprocess
import pandas as pd
import numpy as np

# ================= 配置区域 =================
# 必须和 Notebook 里的设置保持一致
DATASET = 'Grocery_and_Gourmet_Food'
RAW_PATH = os.path.join('..', 'data', DATASET)
DATA_FILE = 'reviews_{}_5.json.gz'.format(DATASET)
META_FILE = 'meta_{}.json.gz'.format(DATASET)

# ================= 1. 下载原始数据 (复用 Cell 47, 49) =================
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

print("Step 1: 检查并下载原始数据...")
if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH)

if not os.path.exists(os.path.join(RAW_PATH, DATA_FILE)):
    print(f'Downloading {DATA_FILE}...')
    subprocess.call(f'cd {RAW_PATH} && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{DATA_FILE}', shell=True)

if not os.path.exists(os.path.join(RAW_PATH, META_FILE)):
    print(f'Downloading {META_FILE}...')
    subprocess.call(f'cd {RAW_PATH} && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{META_FILE}', shell=True)

# ================= 2. 读取并重建 ID 映射=================
print("Step 2: 读取交互数据以重建 Item ID 映射...")
# 读取交互数据
data_df = get_df(os.path.join(RAW_PATH, DATA_FILE))

out_df = data_df.rename(columns={'asin': 'item_id', 'reviewerID': 'user_id', 'unixReviewTime': 'time'})
out_df = out_df[['user_id', 'item_id', 'time']]
out_df = out_df.drop_duplicates(['user_id', 'item_id', 'time'])
out_df = out_df.sort_values(by=['time', 'user_id'], kind='mergesort').reset_index(drop=True)

iids = sorted(out_df['item_id'].unique())
item2id = dict(zip(iids, range(1, len(iids) + 1))) # ASIN -> int ID

print(f"映射重建完成，共有 {len(item2id)} 个物品。")

# ================= 3. 提取文本信息 =================
print("Step 3: 读取元数据并提取文本标题和类别...")
meta_df = get_df(os.path.join(RAW_PATH, META_FILE))

# 过滤掉没用过的物品
useful_meta_df = meta_df[meta_df['asin'].isin(data_df['asin'])].reset_index(drop=True)
enriched_data = []

for idx, row in useful_meta_df.iterrows():
    asin = row['asin']
    if asin in item2id:
        mapped_id = item2id[asin]
        title = row.get('title', 'Unknown Item')
    
        cate_str = "General Grocery"
        categories = row.get('categories', [])
        if len(categories) > 0 and len(categories[0]) > 2:
        
            cate_str = categories[0][2]
        elif len(categories) > 0 and len(categories[0]) > 0:
            cate_str = categories[0][-1] # 否则取最后一个
            
        enriched_data.append({
            'item_id': mapped_id,
            'title': title,
            'i_category': cate_str 
        })

# ================= 4. 保存新文件 =================
print("Step 4: 保存为 item_meta_enriched.csv ...")
enriched_df = pd.DataFrame(enriched_data)
# 按照 item_id 排序
enriched_df = enriched_df.sort_values('item_id')

save_path = os.path.join(RAW_PATH, 'item_meta_enriched.csv')
enriched_df.to_csv(save_path, sep='\t', index=False)

print(f"✅ 成功！已生成包含文本的文件: {save_path}")
print(enriched_df.head())