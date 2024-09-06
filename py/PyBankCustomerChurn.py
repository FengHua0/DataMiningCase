from dataset_process import load_data, split_data
from preprocessing import preprocessing

def main():
    # 进行预处理并保存数据到文件
    data_path = '../data/data.csv'
    df = load_data(data_path)
    preprocessing(df)

    # 划分数据集
    processed_data_path = '../data/processed_data.csv'
    df_processed = load_data(processed_data_path)
    split_data(df_processed)

    #进行过采样
    train_path = '../data/train.csv'
    df_train = load_data(train_path)

if __name__ == '__main__':
    main()
