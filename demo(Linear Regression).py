import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# 读取数据
df = pd.read_csv('data.csv', encoding='ISO-8859-1', on_bad_lines='skip')

# 查看数据的前几行
print(df.head())

# 分离特征和目标变量
X = df.drop(columns=['target'])  # 替换 'target_column_name' 为目标变量的列名
y = df['target']  # 替换 'target_column_name' 为目标变量的列名

# 标准化特征数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy:.2f}")


