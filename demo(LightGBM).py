import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 完整路径读取数据集
data = pd.read_csv('data.csv')

# 假设最后一列是目标变量，其他列是特征
X = data.iloc[:, :-1].values  # 特征数据
y = data.iloc[:, -1].values   # 目标数据

# 拆分为训练集和测试集，70%的数据用于训练，30%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#标准化
transfer = StandardScaler()
X_train = transfer.fit_transform(X_train)
X_test = transfer.transform(X_test)

# 创建LightGBM分类器
model = lgb.LGBMClassifier()
# 训练模型
model.fit(X_train, y_train)
# 进行预测
y_pred = model.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'模型准确率: {accuracy:.2f}')

from sklearn.metrics import f1_score
print('f1_score:')
print(f1_score(y_test,y_pred,average = 'macro'))
