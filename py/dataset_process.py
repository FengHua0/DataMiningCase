import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """加载CSV文件到DataFrame."""
    return pd.read_csv(file_path)


def split_data(df, test_size=0.20, val_size=0.10):
    """
    将数据分为训练集、验证集和测试集。

    参数:
    df -- 输入的完整DataFrame
    test_size -- 测试集所占的比例（相对于整个数据集）
    val_size -- 验证集所占的比例（相对于训练集后的数据）

    返回:
    train_data, val_data, test_data -- 分割后的训练集、验证集和测试集
    """
    # 先划分出测试集
    train_val_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

    # 计算验证集大小的正确比例
    # 因为验证集是从剩余的训练集中划分出来的
    valid_test_size = val_size / (1 - test_size)

    # 划分出验证集
    train_data, val_data = train_test_split(train_val_data, test_size=valid_test_size, random_state=42)

    return train_data, val_data, test_data

#测试用
if __name__ == "__main__":
    # 指定数据文件路径
    data_path = '../data/data.csv'

    # 加载数据
    df = load_data(data_path)

    # 划分数据集
    train_data, val_data, test_data = split_data(df)

    # 显示结果
    print("训练集大小:", train_data.shape)
    print("验证集大小:", val_data.shape)
    print("测试集大小:", test_data.shape)

    # 保存数据集到CSV文件，如果需要
    train_data.to_csv('../data/train.csv', index=False)
    val_data.to_csv('../data/val.csv', index=False)
    test_data.to_csv('../data/test.csv', index=False)
