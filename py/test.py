import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 创建样本数据
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 划分数据
train_x, vali_x, train_y, vali_y = train_test_split(X, y, test_size=0.33, random_state=1234)

# 创建 LGBMClassifier
clf = lgb.LGBMClassifier(
    max_depth=20,
    min_data_in_bin=5,
    max_bin=200,
    min_child_samples=90,
    num_leaves=20,
    n_estimators=20000,
    objective='binary',
    boosting_type='gbdt',
    learning_rate=0.02,
    lambda_l2=5
)

# 检查模型实例类型
print(type(clf))
print("LightGBM version:", lgb.__version__)

# 训练模型，使用 early_stopping_rounds
clf.fit(train_x, train_y, eval_set=[(vali_x, vali_y)], eval_metric='f1', early_stopping_rounds=100)
