import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    加载单个数据文件。
    """
    df = pd.read_csv(file_path)
    return df

def preprocessing(df, feats_list=0, label='Attrition_Flag', id_1='CLIENTNUM'):
    """
    对数据集进行预处理，并将处理后的数据保存到CSV文件。
    """
    print("=============开始处理数据集===============")
    print(f"未处理数据集大小：", df.shape)

    # 检查指定列（如 CLIENTNUM）上的重复并删除
    initial_count = df.shape[0]
    df.drop_duplicates(subset=[id_1], inplace=True)  # 指定subset参数以仅在CLIENTNUM列检查重复
    duplicates_removed = initial_count - df.shape[0]
    print(f"从数据集中移除了{duplicates_removed}条在{id_1}列重复的数据。")

    try:
        # 删除包含缺失值的行
        df.dropna(subset=[id_1, 'Customer_Age', 'Dependent_count', label], inplace=True)

        # 保留特定列，并确保这些列的数值为非负数
        valid_columns = ['Dependent_count', 'Customer_Age', 'Months_on_book', 'Total_Relationship_Count',
                         'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                         'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
                         'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                         'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                         'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
        df = df[(df[valid_columns] >= 0).all(axis=1)].reset_index(drop=True)
    except KeyError as e:
        print(f"数据集里没有相应字段: {e}")

    # 将 'Attrition_Flag' 列中的 'Existing Customer' 替换为 1, 'Attrited Customer' 替换为 0
    if label in df.columns:
        df[label] = df[label].replace({'Existing Customer': 1, 'Attrited Customer': 0})

    # 分离标签列
    y = df[label]
    df.drop(columns=[label], inplace=True)

    # 转换字符类型数据为数值类型
    non_numeric_columns = df.select_dtypes(include=['object', 'category']).columns
    if not non_numeric_columns.empty:
        print(f"检测到非数值列: {non_numeric_columns.tolist()}")
        df = pd.get_dummies(df, columns=non_numeric_columns, drop_first=True)  # 使用独热编码处理非数值列

    # 转换布尔值 TRUE 和 FALSE 为 1 和 0
    bool_columns = df.select_dtypes(include=['bool']).columns
    if not bool_columns.empty:
        print(f"将布尔列转换为数值: {bool_columns.tolist()}")
        df[bool_columns] = df[bool_columns].astype(int)

    print(f"处理后数据集大小：", df.shape)

    if feats_list == 0:
        print(f"使用全部特征")
        X = df
    elif type(feats_list) == list:
        print(f"使用列表特征，长度为：", len(feats_list))
        X = df[feats_list]
    else:
        print("feats_list输入有误")
        return

    X.fillna(0, inplace=True)
    df_final = X.copy()
    df_final[label] = y  # 重新将标签列添加回数据

    # 保存处理后的数据
    file_name = "../data/processed_data.csv"
    df_final.to_csv(file_name, index=False)
    print(f"数据已保存到 {file_name}")

    print("==============数据处理和保存完成=================")

# 仅测试用
if __name__ == '__main__':
    # 指定数据文件路径
    data_path = '../data/data.csv'

    # 加载数据
    df = load_data(data_path)

    # 进行预处理并保存数据到文件
    preprocessing(df)
