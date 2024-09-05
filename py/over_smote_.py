import lightgbm as lgb  # 模型
import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
from sklearn import metrics  # 混淆句子
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split  # 分层五折验证包、寻找最优参函数、切分数据
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix  # 准确率、roc计算、auc计算、混淆矩阵
import itertools  # 处理混淆矩阵
import gc  # 处理缓存，有兴趣的可以搜搜怎么使用
import warnings  # 忽略普通警告，不打印太多东西
from sklearn.feature_selection import RFE, RFECV  # 递归消除选特征，前者是自己选优化到多少位，后者是自动cv优化到最佳
from imblearn.under_sampling import RandomUnderSampler  # 朴素随机过采样，由于是比较旧的这里不做例子
from imblearn.over_sampling import SMOTE, ADASYN  # 目前流行的过采样

warnings.filterwarnings('ignore')

def over_smote_(X, y, num):
    """
    功能: 二分类过采样，以smote举例。
    why: 当正负样本比例相差过大时，一般为1：20以内。举例：如果正负样本为1：99，那么相当于告诉模型只要判断为负，则正确率就为99%，那么模型就会这么做。
    X: 数据X（df型/无label）
    y: 数据y（df型/label）
    num: 过采样的个数
    return:
        X_resampled: 过采样后的X
        y_resampled: 过采样后的y
    """
    ss = pd.Series(y).value_counts()
    smote = SMOTE(sampling_strategy={0: ss[0], 1: ss[1] + num}, random_state=2019)  # radom_state为随机值种子，1:ss[1]+表示label为1的数据增加多少个
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("过采样个数为：", num)
    check_num_X = X_resampled.shape[0] - X.shape[0]
    check_num_y = y_resampled.shape[0] - y.shape[0]
    if (check_num_X == check_num_y) and (check_num_X == num):
        print("过采样校验：成功")
        return X_resampled, y_resampled
    else:
        print("过采样校验：失败")

def main():
    # 加载处理好的训练数据
    train_path = '../data/processed_train.csv'
    df_train = pd.read_csv(train_path)

    # 假设 'Attrition_Flag' 是标签列
    try:
        X_train = df_train.drop('Attrition_Flag', axis=1)  # 提取特征数据
        y_train = df_train['Attrition_Flag']  # 提取标签数据
    except KeyError as e:
        print(f"错误：找不到列 {e}。请检查数据文件的列名是否正确。")
        return

    # 确保所有特征是数值型
    non_numeric_columns = X_train.select_dtypes(include=['object', 'category']).columns
    if not non_numeric_columns.empty:
        print(f"检测到非数值列: {non_numeric_columns.tolist()}")
        X_train = pd.get_dummies(X_train, columns=non_numeric_columns, drop_first=True)

    # 定义需要增加的少数类样本数量
    num_to_increase = 500  # 假设需要增加500个少数类样本

    # 应用 SMOTE 进行过采样
    X_resampled, y_resampled = over_smote_(X_train, y_train, num_to_increase)

    # 输出结果
    print(f"原始训练集样本数: {X_train.shape[0]}, 过采样后训练集样本数: {X_resampled.shape[0]}")
    print("类别分布：")
    print(y_resampled.value_counts())

if __name__ == "__main__":
    main()
