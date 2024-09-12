import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier  # 导入GradientBoostingClassifier
from sklearn.feature_selection import RFECV  # 导入RFECV用于递归特征消除

def rfecv_(X, y, feats, cv=5, scoring='roc_auc'):
    """
    功能: 减少特征，递归消除选特征，输出结果最优最少的特征组。基于GBC模型
    why: 防止特征冗余，该方法有一定的正反性，即最佳的特征组可能是当前数据的最近，以后数据变化了可能就不是了，建议多测几次。
    X: 训练数据X（无标签/df型）
    y: 训练数据y（标签/df型）
    feats: 特征集（list性/一般是去掉id和label），可用该方法生成 feats = [x for x in data.columns if x not in ['id','label']]
    return:
        rfe_cv_model: 特征相关信息对象
        selected_feat: 当前数据消除后特征组
    """
    gbc_model = GradientBoostingClassifier(
        max_depth=3,  # 控制树的最大深度，较小的深度可以加快训练速度
        n_estimators=100,  # 树的数量减少，控制训练时间
        learning_rate=0.1,  # 学习率适当调高，减少树的数量
        min_samples_split=100,  # 增加内部节点再划分所需最小样本数，减少过拟合
        min_samples_leaf=50,  # 增加叶子节点最少样本数
        random_state=1234  # 保持结果可重复性
    )
    print("================开始挑选特征================")
    rfe_cv_model = RFECV(gbc_model, cv=cv, scoring=scoring, verbose=0)  # 使用RFECV进行递归特征消除
    rfe_cv_model.fit(X, y)  # 训练模型
    selected_feat = np.array(feats)[rfe_cv_model.support_].tolist()  # 获取选择的特征
    print("剩余特征：", len(selected_feat))
    print("剩余特征为：")
    for feat in selected_feat:
        print(feat)
    selected_feat_df = pd.DataFrame(selected_feat, columns=['Selected Features'])  # 将选择的特征存为DataFrame
    return rfe_cv_model, selected_feat_df
