from dataset_process import load_data, split_data
from preprocessing import preprocessing

def main():
    # 指定数据文件路径
    data_path = '../data/data.csv'

    # 加载数据
    df = load_data(data_path)

    # 划分数据集
    train_data, val_data, test_data = split_data(df)

    # 进行预处理并保存数据到文件
    preprocessing(train_data, test_data, val_data)

if __name__ == '__main__':
    main()
