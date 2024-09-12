import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
from sklearn.ensemble import GradientBoostingClassifier  # 使用GBC替代LightGBM
from sklearn.model_selection import StratifiedKFold  # 分层五折验证包
from sklearn.metrics import roc_curve, auc, confusion_matrix  # 准确率、roc计算、auc计算、混淆矩阵
import joblib  # 引入joblib库用于保存模型
import warnings  # 忽略普通警告，不打印太多东西
warnings.filterwarnings('ignore')


def train_5_cross(df_pre, X, y, X_test_v1, y_test_v1, thresholds=0.45, id_1='CLIENTNUM', csv_name=0):
    # 定义保存指标的变量
    vali_auc_num = 0  # 验证集AUC
    vali_recall_num = 0  # 验证集召回率
    vali_precision_num = 0  # 验证集精确率
    test_auc_num = 0  # 预测集AUC
    test_recall_num = 0  # 预测集召回率
    test_precision_num = 0  # 预测集精确率
    y_pred_input = np.zeros(len(X_test_v1))  # 相应大小的零矩阵

    print("=============开始训练================")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)  # 分层采样, n_splits为几折

    # 初始化最终模型
    final_model = None

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print("第 {} 次训练...".format(fold_ + 1))
        train_x, trai_y = X.loc[trn_idx], y.loc[trn_idx]
        vali_x, vali_y = X.loc[val_idx], y.loc[val_idx]

        # 使用梯度提升分类器
        clf = GradientBoostingClassifier(
            max_depth=3,  # 减少树的最大深度
            n_estimators=200,  # 减少树的数量
            learning_rate=0.1,  # 增加学习率
            min_samples_split=100,  # 增加内部节点再划分所需最小样本数
            min_samples_leaf=50,  # 增加叶子节点最少样本数
            subsample=0.8,  # 使用部分数据进行训练
            random_state=1234,  # 随机种子
            n_iter_no_change=10,  # 早停条件
            validation_fraction=0.1  # 使用10%的数据作为验证集进行早停
        )
        clf.fit(train_x, trai_y)

        # 更新最终模型为当前训练的模型
        final_model = clf

        # 验证集AUC
        y_prb = clf.predict_proba(vali_x)[:, 1]
        fpr, tpr, thres = roc_curve(vali_y, y_prb)
        vali_roc_auc = auc(fpr, tpr)
        vali_auc_num += vali_roc_auc
        print("vali auc = {0:.4}".format(vali_roc_auc))

        # 预测集AUC
        y_prb_test = clf.predict_proba(X_test_v1)[:, 1]
        fpr, tpr, thres = roc_curve(y_test_v1, y_prb_test)
        test_roc_auc = auc(fpr, tpr)
        test_auc_num += test_roc_auc
        print("test auc = {0:.4}".format(test_roc_auc))

        # 验证metric操作
        y_predictions = y_prb > thresholds
        cnf_matrix = confusion_matrix(vali_y, y_predictions)
        np.set_printoptions(precision=2)
        vali_recall = '{0:.3f}'.format(cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
        vali_precision = '{0:.3f}'.format(cnf_matrix[1, 1] / (cnf_matrix[0, 1] + cnf_matrix[1, 1]))
        print("vali_metric: ", vali_recall, vali_precision)
        vali_recall_num += float(vali_recall)
        vali_precision_num += float(vali_precision)

        # 预测metric操作
        y_predictions_test = y_prb_test > thresholds
        cnf_matrix_test = confusion_matrix(y_test_v1, y_predictions_test)
        test_recall = '{0:.3f}'.format(cnf_matrix_test[1, 1] / (cnf_matrix_test[1, 0] + cnf_matrix_test[1, 1]))
        test_precision = '{0:.3f}'.format(cnf_matrix_test[1, 1] / (cnf_matrix_test[0, 1] + cnf_matrix_test[1, 1]))
        print("test_metric: ", test_recall, test_precision)
        test_recall_num += float(test_recall)
        test_precision_num += float(test_precision)
        y_pred_input += y_prb_test

    print("5折泛化，验证集AUC：{0:.3f}".format(vali_auc_num / 5))
    print("5折泛化，预测集AUC：{0:.3f}".format(test_auc_num / 5))
    print("5折泛化，验证集recall：{0:.3f}".format(vali_recall_num / 5))
    print("5折泛化，验证集precision：{0:.3f}".format(vali_precision_num / 5))
    print("5折泛化，预测集recall：{0:.3f}".format(test_recall_num / 5))
    print("5折泛化，预测集precision：{0:.3f}".format(test_precision_num / 5))

    # 输出名单
    y_pred_input_end = y_pred_input / 5
    y_pred_input_precision = y_pred_input_end > thresholds
    submission = pd.DataFrame({"CLIENTNUM": df_pre[id_1],
                               "概率": y_pred_input_end,
                               "高精确": y_pred_input_precision})
    if csv_name != 0:
        submission.to_csv(f"{csv_name}预测名单.csv", index=False)
    print("================输出名单名单==================")
    print(submission.head(5))

    # 保存最终模型
    final_model_filename = '../data/final_gbc_model_5_fold.joblib'
    joblib.dump(final_model, final_model_filename)
    print(f"最终模型已保存至 {final_model_filename}")

    return final_model
