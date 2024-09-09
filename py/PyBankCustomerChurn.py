import numpy as np

from dataset_process import load_data, split_data
from preprocessing import preprocessing
from over_smote_ import over_smote_
from model_fusion import model_fusion
from py.just_num_leaves import just_num_leaves
from py.rfecv_ import rfecv_
import lightgbm as lgb

from py.train_2_cross import train_2_cross
from py.train_5_cross import train_5_cross


def main():
    # 数据预处理
    data_path = '../data/data.csv'
    df = load_data(data_path)
    preprocessing(df)

    # 数据集划分
    processed_data_path = '../data/processed_data.csv'
    df_processed = load_data(processed_data_path)

    # 过采样
    try:
        X_train = df_processed.drop('Attrition_Flag', axis=1)
        y_train = df_processed['Attrition_Flag']
    except KeyError as e:
        print(f"错误：找不到列 {e}。请检查数据文件的列名是否正确。")
        return

    # 计算过采样数量并进行过采样
    counts = y_train.value_counts()
    num_to_increase = int(abs(counts[1] - counts[0]))
    X_resampled, y_resampled = over_smote_(X_train, y_train, num_to_increase)
    # X_resampled.to_csv('../data/X_resampled.csv', index=False)
    # y_resampled.to_csv('../data/y_resampled.csv', index=False)
    # print(y_resampled.value_counts(0))

    # 特征选择
    lgb_model = lgb.LGBMClassifier(
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        min_gain_to_split=0.01,
        learning_rate=0.05,
        n_estimators=1000,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l1=0.1,
        lambda_l2=0.1,
        objective='binary',
        boosting_type='gbdt',
        metric='auc',
        verbosity=-1
    )
    # 假设 'CLIENTNUM' 是需要排除的列
    feats = [x for x in X_resampled.columns if x != 'CLIENTNUM']  # 从列名列表中去掉 'CLIENTNUM'
    X_noCLIENTNUM = X_resampled[feats]  # X_noCLIENTNUM 仅包含 feats 列表中的特征
    _, train_selected_feats = rfecv_(X_noCLIENTNUM, y_resampled, feats, lgb_model, cv=2)

    # train_selected_feats.to_csv('../data/train_selected_feats.csv', index=False)

    # 参数调优
    selected_feature_names = train_selected_feats['Selected Features'].tolist()
    train_selected = X_resampled[selected_feature_names]
    # train_selected.to_csv('../data/train_selected.csv', index=False)
    num_leave = just_num_leaves(train_selected, y_resampled, start_num=10, end_num=100, step=10, cv=2)

    # 加载测试数据并确保特征一致
    y_test = df_processed['Attrition_Flag']
    X_test = df_processed.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1)  # 确保去掉与训练集一致的列
    X_test = X_test[selected_feature_names]  # 使用选定的特征进行测试

    # # debugs
    # print(f"train_selected shape: {train_selected.shape}")
    # print(f"train_y_resampled shape: {y_resampled.shape}")
    #
    # 二折交叉验证
    clf = train_2_cross(df_processed, train_selected, y_resampled, X_test, y_test, thresholds=0.45, csv_name='2', num_leave=num_leave)

    # # 五折交叉验证
    # clf = train_5_cross(df_processed, train_selected, y_resampled, X_test, y_test, thresholds=0.45, csv_name='5', num_leave=num_leave)

if __name__ == '__main__':
    main()
