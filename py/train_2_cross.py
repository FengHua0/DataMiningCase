import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
from sklearn.ensemble import GradientBoostingClassifier  # 使用GBC替代LightGBM
from sklearn.model_selection import train_test_split  # 切分数据
from sklearn.metrics import roc_curve, auc, confusion_matrix  # 准确率、roc计算、auc计算、混淆矩阵
import joblib  # 引入joblib库用于保存模型
import warnings  # 忽略普通警告，不打印太多东西

warnings.filterwarnings('ignore')


def train_2_cross(df_pre, X, y, X_test_v1, y_test_v1, thresholds=0.45, id_1='CLIENTNUM', csv_name=0):
    """
    功能: 切分一次训练，输出名单
    why: 两折一般是上线的版本。因为比较简单直接
    X: 训练数据X（无标签/df型）
    y: 训练数据y（标签/df型）
    X_test_v1: 预测数据X（无标签/df型）
    y_test_v1: 预测数据y（无标签/df型）
    thresholds: 阈值选择，默认0.45高精确率
    csv_name: 保存csv的名称，默认不保存
    return:
        客户名单及情况
        clf: 已训练好的模型
    """
    y_pred_input = np.zeros(len(X_test_v1))  # 相应大小的零矩阵
    train_x, vali_x, train_y, vali_y = train_test_split(X, y, test_size=0.33, random_state=1234)
    print("================开始二折交叉验证================")

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
    clf.fit(train_x, train_y)

    # ===============验证集AUC操作===================
    y_prb = clf.predict_proba(vali_x)[:, 1]  # 获取预测概率
    fpr, tpr, thres = roc_curve(vali_y, y_prb)
    vali_roc_auc = auc(fpr, tpr)  # 获取验证集auc
    print("vali auc = {0:.4}".format(vali_roc_auc))  # 本次auc的值

    # ===============预测集AUC操作===================
    y_prb_test = clf.predict_proba(X_test_v1)[:, 1]  # 获取预测概率
    fpr, tpr, thres = roc_curve(y_test_v1, y_prb_test)
    test_roc_auc = auc(fpr, tpr)
    print("test auc = {0:.4}".format(test_roc_auc))

    # ===============训练metric操作===================
    y_predictions = y_prb > thresholds  # 取阈值多少以上的为True
    cnf_matrix = confusion_matrix(vali_y, y_predictions)  # 建立矩阵
    np.set_printoptions(precision=2)  # 控制在两位数
    vali_recall = '{0:.3f}'.format(cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))  # 召回率
    vali_precision = '{0:.3f}'.format(cnf_matrix[1, 1] / (cnf_matrix[0, 1] + cnf_matrix[1, 1]))  # 精确率
    print("vali_metric: ", vali_recall, vali_precision)

    # ===============预测metric操作===================
    y_predictions_test = y_prb_test > thresholds  # 取阈值多少以上的为True
    cnf_matrix_test = confusion_matrix(y_test_v1, y_predictions_test)  # 建立矩阵
    test_recall = '{0:.3f}'.format(cnf_matrix_test[1, 1] / (cnf_matrix_test[1, 0] + cnf_matrix_test[1, 1]))  # 召回率
    test_precision = '{0:.3f}'.format(cnf_matrix_test[1, 1] / (cnf_matrix_test[0, 1] + cnf_matrix_test[1, 1]))  # 精确率
    print("test_metric: ", test_recall, test_precision)

    print("================开始输出名单==================")
    y_pred_input_precision = y_prb_test > thresholds  # 获取高精确率的标签
    submission = pd.DataFrame({"CLIENTNUM": df_pre[id_1],
                               "概率": y_prb_test,
                               "高精确": y_pred_input_precision})
    if csv_name != 0:
        submission.to_csv("%s预测名单.csv" % csv_name, index=False)  # 保存
    print("================输出名单名单==================")
    print(submission.head(5))

    # 保存最终模型
    final_model_filename = '../data/final_gbc_model_2_fold.joblib'
    joblib.dump(clf, final_model_filename)
    print(f"最终模型已保存至 {final_model_filename}")

    return clf
