import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as ex
# import plotly.graph_objs as go
# import plotly.figure_factory as ff
# from plotly.subplots import make_subplots
# import plotly.offline as pyo
# pyo.init_notebook_mode()
sns.set_style('darkgrid')
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix

def feature_create(customers_data):
    """

    客户年龄    Customer_Age
    性别       Gender
    教育水平    Education_Level
    婚姻状况    Marital_Status
    收入水平    income level
    婚姻状况    marital status
    信用卡类型  Card_Category
    依赖计数    Dependent_count
    信用额度    Credit_Limit
    每年活动业务数 Months_Inactive_12_mon

    """
    customers_data = customers_data[customers_data.columns[:-2]]# 取所有列但去掉最后两列
    print(customers_data)

    print("缺失值个数: ", customers_data.duplicated().sum())
    print("重复值个数:\n", customers_data.isna().sum())
    print("数据集信息:\n", customers_data.info())

    # 设置显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 客户流失与年龄的关系
    x1 = customers_data['Attrition_Flag']
    y1 = customers_data['Customer_Age']
    print("总客户数:", customers_data.shape[0])
    print("流失客户数:", x1.sum())
    plt.boxplot([y1[x1 == 0], y1[x1 == 1]])
    plt.xticks([1, 2], ['0', '1'])  # 指定横坐标位置和对应标签
    plt.xlabel('客户流失情况')
    plt.ylabel('客户年龄')
    plt.title('客户流失情况与年龄关系图')
    plt.show()

    # 客户流失与性别的关系
    gender_group = customers_data.groupby('Gender').agg({'Attrition_Flag': ['sum', 'count']})# 使用聚合函数对每个组进行操作。这里针对“Attrition_Flag”列分别计算“sum”（总和）和“count”（数量）
    gender_group['Churn_Rate'] = gender_group[('Attrition_Flag', 'sum')] / gender_group[('Attrition_Flag', 'count')]# 计算每个性别组的流失率。通过用“Attrition_Flag”列的总和除以数量，得到流失率的值
    print(gender_group)
    plt.bar(gender_group.index, gender_group['Churn_Rate'])
    plt.xlabel('性别')
    plt.ylabel('客户流失率')
    plt.title('客户流失情况与性别关系图')
    plt.show()

    # 客户流失与教育水平的关系
    edu_group = customers_data.groupby('Education_Level').agg({'Attrition_Flag': ['sum', 'count']})
    edu_group['Churn_Rate'] = edu_group[('Attrition_Flag', 'sum')] / edu_group[('Attrition_Flag', 'count')]
    print(edu_group)
    plt.bar(edu_group.index, edu_group['Churn_Rate'])
    plt.xlabel('教育水平')
    plt.ylabel('客户流失率')
    plt.title('客户流失情况与教育水平关系图')
    plt.show()

    # 客户流失与婚姻情况的关系
    maried_group = customers_data.groupby('Marital_Status').agg({'Attrition_Flag': ['sum', 'count']})
    maried_group['Churn_Rate'] = maried_group[('Attrition_Flag', 'sum')] / maried_group[('Attrition_Flag', 'count')]
    print(maried_group)
    plt.bar(maried_group.index, maried_group['Churn_Rate'])
    plt.xlabel('婚姻情况')
    plt.ylabel('客户流失率')
    plt.title('客户流失情况与婚姻情况关系图')
    plt.show()

    # 客户流失与信用卡类型的关系
    card_group = customers_data.groupby('Card_Category').agg({'Attrition_Flag': ['sum', 'count']})
    card_group['Churn_Rate'] = card_group[('Attrition_Flag', 'sum')] / card_group[('Attrition_Flag', 'count')]
    print(card_group)
    plt.bar(card_group.index, card_group['Churn_Rate'])
    plt.xlabel('信用卡类型')
    plt.ylabel('客户流失率')
    plt.title('客户流失情况与信用卡类型关系图')
    plt.show()

    # 客户流失与依赖计数的关系
    dep_group = customers_data.groupby('Dependent_count').agg({'Attrition_Flag': ['sum', 'count']})
    dep_group['Churn_Rate'] = dep_group[('Attrition_Flag', 'sum')] / dep_group[('Attrition_Flag', 'count')]
    print(dep_group)
    plt.bar(dep_group.index, dep_group['Churn_Rate'])
    plt.xlabel('依赖计数')
    plt.ylabel('客户流失率')
    plt.title('客户流失情况与依赖计数关系图')
    plt.show()

    # 客户流失情况与信用额度限制的关系
    y2 = customers_data['Credit_Limit']
    plt.boxplot([y2[x1 == 0], y2[x1 == 1]])
    plt.xticks([1, 2], ['0', '1'])  # 指定横坐标位置和对应标签
    plt.xlabel('客户流失情况')
    plt.ylabel('客户信用额度限制')
    plt.title('客户流失情况与信用额度限制关系图')
    plt.show()

    # 客户流失情况与每年业务活动数的关系
    y3 = customers_data['Months_Inactive_12_mon']
    plt.boxplot([y3[x1 == 0], y3[x1 == 1]])
    plt.xticks([1, 2], ['0', '1'])  # 指定横坐标位置和对应标签
    plt.xlabel('客户流失情况')
    plt.ylabel('客户每年业务活动数')
    plt.title('客户流失情况与每年业务活动数关系图')
    plt.show()

    # 客户流失情况与每月预订数的关系
    y4 = customers_data['Months_on_book']
    plt.boxplot([y4[x1 == 0], y4[x1 == 1]])
    plt.xticks([1, 2], ['0', '1'])  # 指定横坐标位置和对应标签
    plt.xlabel('客户流失情况')
    plt.ylabel('客户每月预订数')
    plt.title('客户流失情况与每月预订数关系图')
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('../data/data.csv')
    feature_create(data)