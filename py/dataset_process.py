import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(file_path):
    """加载CSV文件到DataFrame."""
    return pd.read_csv(file_path)

def split_data(df, test_size=0.10):
    """
    将数据分为训练集和测试集。

    参数:
    df -- 输入的完整DataFrame
    test_size -- 测试集所占的比例（相对于整个数据集）

    返回:
    train_data, test_data -- 分割后的训练集和测试集
    """
    # 转换字符类型数据为数值类型
    non_numeric_columns = df.select_dtypes(include=['object', 'category']).columns
    if not non_numeric_columns.empty:
        print(f"检测到非数值列: {non_numeric_columns.tolist()}")
        # 使用LabelEncoder进行标签编码
        label_encoder = LabelEncoder()
        for col in non_numeric_columns:
            df[col] = label_encoder.fit_transform(df[col])

    df.to_csv('../data/processed_data.csv', index=False)
    # 划分出测试集
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

    # 保存训练集和测试集到CSV文件
    train_data.to_csv('../data/train.csv', index=False)
    test_data.to_csv('../data/test.csv', index=False)

    return train_data, test_data