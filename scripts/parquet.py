import pandas as pd

# 读取 Parquet 文件
df1 = pd.read_parquet("/home/qiaojun/.cache/huggingface/lerobot/flexiv_boxppfixed_stateq_actioneef/data/chunk-000/episode_000000.parquet")
df2 = pd.read_parquet("/home/qiaojun/.cache/huggingface/lerobot/flexiv_boxppfixed_stateq_actioneef/data/chunk-000/episode_000000.parquet")

# 显示数据
print(df1['action']-df2['action'])