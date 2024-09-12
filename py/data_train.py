import pandas as pd
from PIL._imaging import display
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV


def data_train(df_gdbt):
    target = df_gdbt[['Attrition_Flag']]
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(df_gdbt, target, test_size=0.2, random_state=42)
    # 构建模型
    gbdt = GradientBoostingClassifier()
    # 训练模型
    gbdt.fit(x_train, y_train)
    # 返回各个特征的重要性，数值越大，特征越重要
    importances = gbdt.feature_importances_
    # 将这个特征重要性以图表形式可视化显示
    importances_df = pd.DataFrame(importances, index=df_gdbt.columns, columns=['importance'])
    # 按特征重要性降序排列
    importances_df = importances_df.sort_values(by='importance', ascending=False)
    display(importances_df)

    # 画柱状图
    plt.bar(df_gdbt.columns, importances_df['importance'])
    plt.xlabel('数据特征')
    plt.ylabel('特征重要性')
    plt.show()
    # 在测试集上进行预测
    y_pred = gbdt.predict(x_test)
    print("梯度提升决策树准确度:", gbdt.score(x_test, y_test))
    print("其他指标：\n", classification_report(y_test, y_pred))

    # 创建 GBDT 模型
    gbdt1 = GradientBoostingClassifier()
    parameters = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 4, 5]
    }

    # 创建 GridSearchCV 对象
    y_train1 = y_train.values.ravel()
    grid_search = GridSearchCV(gbdt1, parameters, cv=5)
    grid_search.fit(x_train, y_train1)

    # 获取最佳参数
    best_params = grid_search.best_params_
    print('Best Parameters Found:', best_params)

    best_gbdt = GradientBoostingClassifier(**best_params)
    # 训练模型
    best_gbdt.fit(x_train, y_train)
    # 在测试集上进行预测
    y_pred = best_gbdt.predict(x_test)
    # 在测试集上评估最终模型
    test_score = best_gbdt.score(x_test, y_test)
    print("最优参数下的梯度提升决策树准确度:", test_score)
    print("其他指标：\n", classification_report(y_test, y_pred))

