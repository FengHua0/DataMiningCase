from dataset_process import load_data, split_data
from preprocessing import preprocessing
from over_smote_ import over_smote_
from model_fusion import model_fusion
from py.just_num_leaves import just_num_leaves
from py.rfecv_ import rfecv_
import lightgbm as lgb

from py.train_2_cross import train_2_cross

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
    num_to_increase = abs(counts[1] - counts[0])
    train_X_resampled, train_y_resampled = over_smote_(X_train, y_train, num_to_increase)

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
    feats = train_X_resampled.columns.tolist()
    _, train_selected_feats = rfecv_(train_X_resampled, train_y_resampled, feats, lgb_model)

    # 参数调优
    train_selected = train_X_resampled[train_selected_feats]
    num_leave = just_num_leaves(train_selected, train_y_resampled, start_num=10, end_num=60, step=10)

    # 加载数据
    y_test = df_processed['Attrition_Flag']
    X_test = df_processed.drop('Attrition_Flag', axis=1)

    # 二折交叉验证
    clf = train_2_cross(df_processed, train_selected, train_y_resampled, X_test, y_test, thresholds=0.45, csv_name='final', num_leave=num_leave)

if __name__ == '__main__':
    main()
