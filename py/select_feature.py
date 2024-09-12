import numpy as np
import pandas as pd
from PIL._imaging import display
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils import stats

from py.data_train import data_train


def select_feature(df):

    # 计算皮尔逊相关系数
    num_df = df.select_dtypes(include=np.number)
    corr_mar = num_df.corr()  # 计算数据框各列之间的皮尔逊相关系数
    corr_df = pd.DataFrame({'Corr': corr_mar['Attrition_Flag']}).reset_index().sort_values('Corr',
                                                                                           ascending=False).rename(
        columns={'index': 'Feature'})
    display(corr_df)
    F = corr_df['Feature']
    C = corr_df['Corr']
    plt.bar(F, C)
    plt.xlabel('影响因素')
    plt.ylabel('相关系数')
    plt.title('各影响因素的皮尔逊相关系数')
    plt.show()

    # 卡方检验(分类变量)
    def chi_square(data, features):
        results = pd.DataFrame(columns=['Feature', 'Chi_square', 'P_value'])
        for feature in features:
            crosstab = pd.crosstab(data[feature], data['Attrition_Flag'])  # 使用 pd.crosstab 函数创建特定特征与'Exited'特征的列联表
            # pd.crosstab是pandas库中的一个函数，主要用于创建交叉表（列联表）。
            # 它接受两个或多个数组或序列作为参数，通过对这些数据进行统计计数，生成一个表格来展示不同类别组合出现的频数。
            chi_square, p_value, dof, expected = stats.chi2_contingency(
                crosstab)  # 使用 stats.chi2_contingency 计算卡方值、p 值等
            row = {'Feature': feature, 'Chi_square': chi_square, 'P_value': p_value}
            results = results._append(row, ignore_index=True)
        return results

    features = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    chi_df = chi_square(df, features)
    display(chi_df)

    # 绘制卡方检验值和P值的折线图
    fig, fig_chi = plt.subplots()  # 创建图表，并设置一个主坐标轴对象
    fig_fea = chi_df['Feature']  # 提取横坐标环境因素
    data_chi = chi_df['Chi_square']
    data_p = chi_df['P_value']  # 提取数据
    line1 = fig_chi.plot(fig_fea, data_chi, color='b')  # 绘制卡方检验值的折线
    fig_chi.set_ylabel('卡方检验值', color='b')
    fig_p = fig_chi.twinx()  # 创建共享横坐标的右边纵坐标fig_p
    line2 = fig_p.plot(fig_fea, data_p, color='r')
    fig_p.set_ylabel('P值', color='r')
    plt.xlabel('Features')
    plt.show()

    # 客户细分(K均值聚类算法)
    df_kmeans = df.copy(deep=True)
    df_kmeans['Gender'] = df_kmeans['Gender'].map({'M': 1, 'F': 0})
    df_kmeans['Education_Level'] = df_kmeans['Education_Level'].map(
        {'Unknown': 0, 'Uneducated': 1, 'High School': 2, 'College': 3, 'Graduate': 4, 'Post-Graduate': 5,
         'Doctorate': 6})
    df_kmeans['Marital_Status'] = df_kmeans['Marital_Status'].map(
        {'Unknown': 0, 'Single': 1, 'Married': 2, 'Divorced': 3})
    df_kmeans['Income_Category'] = df_kmeans['Income_Category'].map(
        {'Unknown': 72000, 'Less than $40K': 20000, '$40K - $60K': 50000, '$60K - $80K': 70000, '$80K - $120K': 100000,
         '$120K +': 120000})
    df_kmeans['Card_Category'] = df_kmeans['Card_Category'].map({'Blue': 0, 'Gold': 1, 'Silver': 2, 'Platinum': 3})
    # 选择一些合适的连续变量特征
    features = df_kmeans[
        ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category',
         'Card_Category', 'Months_on_book', 'Months_Inactive_12_mon', 'Credit_Limit']]
    # 进行数据标准化处理，减少数据之间误差
    scaler = StandardScaler()
    scaler_features = scaler.fit_transform(features)
    # 创建聚类算法对象,并训练
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaler_features)
    # 将聚类结果添加到df数据框里
    df_kmeans['Clusters'] = clusters
    display(df_kmeans[['Attrition_Flag', 'Clusters']].head())

    # 对三个客户群体进行分析各其特征和流失率
    clusters_analysis = df_kmeans.groupby(by='Clusters').agg(
        {'Customer_Age': 'mean', 'Dependent_count': 'mean', 'Income_Category': 'mean', 'Months_on_book': 'mean',
         'Months_Inactive_12_mon': 'mean', 'Credit_Limit': 'mean', 'Attrition_Flag': 'mean'}
    ).reset_index().round(2)
    display(clusters_analysis)

    # GradientBoostingClassifier梯度提升算法
    df_gdbt = df_kmeans[df_kmeans.columns[:-1]]
    # 特征重要性分析
    df_gdbt = df_gdbt.drop(labels=['Attrition_Flag'], axis=1)
    display(df_gdbt)
    df_gdbt.info()


    # 训练
    data_train(df_gdbt)