import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def GBCtrain(X_train, y_train, X_test, y_test):

    print("================开始训练================")
    # 初始化和训练模型
    gbc = GradientBoostingClassifier(
        n_estimators=300,  # 树的数量
        learning_rate=0.05,  # 学习率
        max_depth=4,  # 树的最大深度
        subsample=0.8,  # 采样比例
        min_samples_split=10,  # 最小样本分裂数
        min_samples_leaf=4,  # 叶节点最小样本数
        max_features='sqrt',  # 使用的最大特征数
        random_state=42  # 随机种子
    )
    gbc.fit(X_train, y_train)

    # 进行预测
    y_pred = gbc.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
