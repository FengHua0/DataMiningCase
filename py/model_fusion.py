import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

# 全局参数定义
lgb_params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': -1,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 300,  # 特征抽样的随机种子
    'bagging_seed': 3  # 数据抽样的随机种子
    # 'is_unbalance': True,  # 如果需要平衡类别可以启用
    # 'scale_pos_weight': 98145/1855  # 根据类别比例设置权重
}

xgb_params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'lambda': 1,
    'alpha': 0,
    'silent': 1
}

data_seed = 2020


def get_leaderboard_score(test_df, prediction):
    """
    计算并输出模型的 AUC 评分
    :param test_df: 测试集 DataFrame
    :param prediction: 模型预测结果
    """
    label = test_df['Attrition_Flag'].values  # 提取真实标签
    assert len(prediction) == len(label)  # 确保预测结果和真实标签长度相等
    print('stacking auc score: ', roc_auc_score(label, prediction))  # 输出 AUC 评分


def model_fusion(train, test):
    """
    进行模型融合并评估
    :param train: 训练集 DataFrame
    :param test: 测试集 DataFrame
    """
    print("================模型融合训练================")
    print("数据集大小：", train.shape, test.shape)
    train = train.fillna(-999)  # 填充缺失值
    test = test.fillna(-999)
    feats = [x for x in train.columns if x not in ['CLIENTNUM', 'Attrition_Flag']]  # 提取特征列
    X = train[feats].values
    y = train['Attrition_Flag'].values
    X_test = test[feats].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=data_seed)

    blend_train = pd.DataFrame()  # 用于保存第一层模型的训练结果
    blend_test = pd.DataFrame()  # 用于保存第一层模型的测试结果

    # LightGBM 模型训练
    test_pred_lgb = 0
    cv_score_lgb = []
    train_feats = np.zeros(X.shape[0])
    for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print('training fold: ', idx + 1)
        train_x, valid_x = X[train_idx], X[test_idx]
        train_y, valid_y = y[train_idx], y[test_idx]
        dtrain = lgb.Dataset(train_x, train_y, feature_name=feats)
        dvalid = lgb.Dataset(valid_x, valid_y, feature_name=feats)
        model = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=dvalid)
        valid_pred = model.predict(valid_x, num_iteration=model.best_iteration)
        train_feats[test_idx] = valid_pred
        auc_score = roc_auc_score(valid_y, valid_pred)
        print('LightGBM auc score: ', auc_score)
        cv_score_lgb.append(auc_score)
        test_pred_lgb += model.predict(X_test, num_iteration=model.best_iteration)

    test_pred_lgb /= 5
    blend_train['lgb_feat'] = train_feats
    blend_test['lgb_feat'] = test_pred_lgb

    # XGBoost 模型训练
    test_pred_xgb = 0
    cv_score_xgb = []
    train_feats_xgb = np.zeros(X.shape[0])
    for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print('training fold: ', idx + 1)
        train_x, valid_x = X[train_idx], X[test_idx]
        train_y, valid_y = y[train_idx], y[test_idx]
        dtrain = xgb.DMatrix(train_x, train_y, feature_names=feats)
        dvalid = xgb.DMatrix(valid_x, valid_y, feature_names=feats)
        watchlist = [(dvalid, 'eval')]
        model = xgb.train(xgb_params, dtrain, num_boost_round=2000, evals=watchlist,
                          early_stopping_rounds=50, verbose_eval=50)
        valid_pred = model.predict(dvalid)
        train_feats_xgb[test_idx] = valid_pred
        auc_score = roc_auc_score(valid_y, valid_pred)
        print('XGBoost auc score: ', auc_score)
        cv_score_xgb.append(auc_score)
        dtest = xgb.DMatrix(X_test, feature_names=feats)
        test_pred_xgb += model.predict(dtest)

    test_pred_xgb /= 5
    blend_train['xgb_feat'] = train_feats_xgb
    blend_test['xgb_feat'] = test_pred_xgb

    print(blend_train.head(5))  # 查看第一层模型训练集特征
    print(blend_test.head(5))  # 查看第一层模型测试集特征

    # 第二层模型 - 逻辑回归
    lr_model = LogisticRegression()
    lr_model.fit(blend_train.values, y)
    print("逻辑回归模型的特征权重:", lr_model.coef_)
    test_pred_lr = lr_model.predict_proba(blend_test.values)[:, 1]

    # 计算最终评分
    get_leaderboard_score(test, test_pred_lr)
