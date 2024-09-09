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
# SMOTE: 对于少数类样本a, 随机选择一个最近邻的样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本;
# ADASYN: 关注的是在那些基于K最近邻分类器被错误分类的原始样本附近生成新的少数类样本;


def rfecv_(X, y, feats, cv=5, scoring='roc_auc'):
    """
    功能: 减少特征，递归消除选特征，输出结果最优最少的特征组。基于lgb模型
    why: 防止特征冗余，该方法有一定的正反性，即最佳的特征组可能是当前数据的最近，以后数据变化了可能就不是了，建议多测几次。
    X: 训练数据X（无标签/df型）
    y: 训练数据y（标签/df型）
    feats: 特征集（list性/一般是去掉id和label），可用该方法生成 feats = [x for x in data.columns if x not in ['id','label']]
    lgb_model: 模型参数
    reture:
        rfe_cv_model: 特征相关信息对象
        selected_feat: 当前数据消除后特征组
    """
    lgb_model = lgb.LGBMClassifier(
        max_depth=7,  # 控制树的最大深度，防止过拟合
        num_leaves=31,  # 叶子节点数量，较小的值可以防止过拟合
        min_child_samples=20,  # 每个叶子最少样本数，增大可以减少过拟合
        min_gain_to_split=0.01,  # 分裂的最小增益
        learning_rate=0.05,  # 学习率，较大的值配合较少的树
        n_estimators=500,  # 树的数量，控制训练时间，防止过拟合
        feature_fraction=0.8,  # 每次训练选择的特征比例
        bagging_fraction=0.8,  # 每次训练选择的数据比例
        bagging_freq=5,  # bagging的频率
        lambda_l1=1,  # L1正则化
        lambda_l2=1,  # L2正则化
        objective='binary',  # 二分类
        boosting_type='gbdt',  # 梯度提升树
        metric='f1',  # 使用AUC作为评价指标，更适合不平衡数据集
        verbosity=-1,  # 控制输出
    )
    print("================开始挑选特征================")
    rfe_cv_model = RFECV(lgb_model, cv=cv, scoring=scoring, verbose=0)
    rfe_cv_model.fit(X, y)
    selected_feat = np.array(feats)[rfe_cv_model.support_].tolist()
    print("剩余特征：", len(selected_feat))
    print("剩余特征为：")
    for feat in selected_feat:
        print(feat)
    selected_feat_df = pd.DataFrame(selected_feat, columns=['Selected Features'])
    return rfe_cv_model, selected_feat_df