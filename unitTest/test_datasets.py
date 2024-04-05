from datasets import load_dataset, Dataset
import pandas as pd

# 加载数据集
dataset = load_dataset("json", data_files="./../data/sample.json")
dataset2 = load_dataset("json", data_files="./../data/sample2.json")

# 转换为 pandas DataFrame
df1 = pd.DataFrame(dataset['train'])
df2 = pd.DataFrame(dataset2['train'])

# 按照指定列进行合并
merged_df = pd.merge(df1, df2, how='inner', on='id')

# 输出合并后的 DataFrame
print(merged_df)


merged_dataset = Dataset.from_pandas(merged_df)

print(merged_dataset)