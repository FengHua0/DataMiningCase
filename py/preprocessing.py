import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import itertools
import gc
import warnings

warnings.filterwarnings('ignore')

def load_data(train_path, predict_path, val_path):
    """
    加载训练集、预测集和验证集数据。
    """
    df_train = pd.read_csv(train_path)
    df_pre = pd.read_csv(predict_path)
    df_val = pd.read_csv(val_path)
    return df_train, df_pre, df_val

def preprocessing(df_train, df_pre, df_val, feats_list=0, label='Attrition_Flag', id_1='CLIENTNUM'):
    """
    对训练集、预测集和验证集进行预处理，并将处理后的数据保存到CSV文件。
    """
    datasets = {'processed_train': df_train, 'processed_test': df_pre, 'processed_val': df_val}
    for name, df in datasets.items():
        print(f"=============开始处理{name}数据集===============")
        print(f"未处理{name}数据大小：", df.shape)

        # 检查指定列（如 CLIENTNUM）上的重复并删除
        initial_count = df.shape[0]
        df.drop_duplicates(subset=[id_1], inplace=True)  # 指定subset参数以仅在CLIENTNUM列检查重复
        duplicates_removed = initial_count - df.shape[0]
        print(f"从{name}中移除了{duplicates_removed}条在{id_1}列重复的数据。")

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
            print(f"{name}数据集里没有相应字段: {e}")

        # 分离标签列
        y = df[label]
        df.drop(columns=[label], inplace=True)

        # 转换字符类型数据为数值类型
        non_numeric_columns = df.select_dtypes(include=['object', 'category']).columns
        if not non_numeric_columns.empty:
            print(f"检测到非数值列: {non_numeric_columns.tolist()}")
            df = pd.get_dummies(df, columns=non_numeric_columns, drop_first=True)  # 使用独热编码处理非数值列

        print(f"处理后{name}数据大小：", df.shape)

        if feats_list == 0:
            print(f"使用{name}全部特征")
            X = df[df.columns.drop([id_1])]
        elif type(feats_list) == list:
            print(f"使用{name}列表特征，长度为：", len(feats_list))
            X = df[feats_list]
        else:
            print("feats_list输入有误")
            continue

        X.fillna(0, inplace=True)
        df_final = X.copy()
        df_final[label] = y  # 重新将标签列添加回数据

        file_name = f"../data/{name.lower()}.csv"
        df_final.to_csv(file_name, index=False)
        print(f"{name}数据已保存")

    print("==============数据处理和保存完成=================")


# 测试用
if __name__ == '__main__':
    # 指定数据文件路径
    train_path = '../data/train.csv'
    predict_path = '../data/test.csv'
    val_path = '../data/val.csv'

    # 加载数据
    df_train, df_pre, df_val = load_data(train_path, predict_path, val_path)

    # 进行预处理并保存数据到文件
    preprocessing(df_train, df_pre, df_val)
