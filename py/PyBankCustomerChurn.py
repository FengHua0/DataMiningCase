import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset_process import load_data, split_data
from notebook.py_bar import plt_data
from preprocessing import preprocessing
from over_smote_ import over_smote_
from py.auc_plot import auc_plot
from py.just_num_leaves import just_num_leaves
from py.metrics_plot import metrics_plot
from py.model_train import GBCtrain
from py.rfecv_ import rfecv_
import lightgbm as lgb
from py.train_2_cross import train_2_cross
from py.train_5_cross import train_5_cross


def main():
    # 数据预处理
    data_path = '../data/data.csv'
    df = load_data(data_path)
    train_data, test_data = split_data(df)
    dt_train, dt_test, X, y, X_test_v1, y_test_v1 = preprocessing(train_data, test_data)

    # # 数据集划分
    # processed_data_path = '../data/processed_data.csv'
    # total_data = load_data(processed_data_path)
    # df_processed, test_data = split_data(total_data)

    # 过采样
    # 计算过采样数量并进行过采样
    counts = y.value_counts()
    num_to_increase = abs(counts[1] - counts[0])
    if min(counts.iloc[0], counts.iloc[1]) > ((counts.iloc[0] + counts.iloc[1]) / 99):
        num_to_increase = 0
    X_resampled, y_resampled = over_smote_(X, y, num_to_increase)
    # X_resampled.to_csv('../data/X_resampled.csv', index=False)
    # y_resampled.to_csv('../data/y_resampled.csv', index=False)
    # print(y_resampled.value_counts(0))

    # 二折交叉验证参数部分：
    # 特征选择
    # 假设 'CLIENTNUM' 是需要排除的列
    feats = [x for x in X_resampled.columns if x != 'CLIENTNUM']  # 从列名列表中去掉 'CLIENTNUM'
    X_noCLIENTNUM = X_resampled[feats]  # X_noCLIENTNUM 仅包含 feats 列表中的特征
    _, train_selected_feats = rfecv_(X_noCLIENTNUM, y_resampled, feats, cv=2)

    # train_selected_feats.to_csv('../data/train_selected_feats.csv', index=False)

    # 参数调优
    selected_feature_names = train_selected_feats['Selected Features'].tolist()
    train_selected = X_resampled[selected_feature_names]
    # train_selected.to_csv('../data/train_selected.csv', index=False)
    num_leave = just_num_leaves(train_selected, y_resampled, start_num=10, end_num=150, step=10, cv=2)

    # 二折交叉验证参数
    y_test = dt_test['Attrition_Flag']
    X_test = dt_test[selected_feature_names]

    # 二折交叉验证
    clf_2 = train_2_cross(dt_test, train_selected, y_resampled, X_test, y_test, thresholds=0.45, csv_name='2', num_leave=num_leave)

    # 五折交叉验证参数
    _, train_selected_feats = rfecv_(X_noCLIENTNUM, y_resampled, feats, cv=5)
    num_leave = just_num_leaves(train_selected, y_resampled, start_num=10, end_num=150, step=10, cv=5)
    #debugs
    # 五折交叉验证
    clf_5 = train_5_cross(dt_test, train_selected, y_resampled, X_test, y_test, thresholds=0.45, csv_name='5', num_leave=num_leave)

    # ROC曲线
    prob = clf_5.predict_proba(train_selected)[:, 1]  # 获取概率
    plt_data.auc_plot(y_resampled, prob)  # 查看auc
    metrics_plot(y_resampled, prob, thres=0.45)

    # 混淆矩阵
    plt_data.metrics_plot(y_resampled, prob, thres=0.45)  # 查看混淆矩阵

    most_important_feat = plt_data.importance_plt(train_selected, clf_5)  # 查看特征重要性，决定是否去掉处理特征

    plt_data.kde_plt(train_data, most_important_feat, 'Attrition_Flag')  # 查看最高特征的正负样本区分情况
    plt_data.bar_plt('Attrition_Flag', most_important_feat, train_data)  # 查看最高特征的正负样本区分情况

    # GBC模型训练
    GBCtrain(X, y, X_test_v1, y_test_v1)

if __name__ == '__main__':
    main()
