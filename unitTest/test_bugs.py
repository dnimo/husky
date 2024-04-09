import pandas as pd
from datasets import Dataset

# 示例数据，其中 'Column 1' 包含结构数组
data = {'Column 1': [{'field1': [1, 2], 'field2': [3, 4]},
                     {'field1': [5, 6, 7], 'field2': [8, 9]},
                     {'field1': [11, 12], 'field2': [13, 14]}],
        'Column 2': [1, 2, 3]}

df = pd.DataFrame(data)

# 找到 'Column 1' 中结构数组长度不一致的行
problematic_rows = df[df['Column 1'].apply(lambda x: len(x['field1'])) != df['Column 1'].apply(lambda x: len(x['field2']))]

# 打印问题行
print("Problematic Rows:")
print(problematic_rows)

# 处理长度不一致的行，例如删除这些行
# df = df.drop(index=problematic_rows.index)

# 将 Pandas DataFrame 转为 Dataset 对象
dataset = Dataset.from_pandas(df)

# 打印 Dataset
print(dataset)
