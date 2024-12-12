import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

# 加载数据
data = pd.read_csv("data.csv")

# 数据预处理
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义神经网络模型
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=42)

# 训练模型
mlp_model.fit(X_train, y_train)

# 预测
y_pred_mlp = mlp_model.predict(X_test)

# 评估模型
f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')
print(f'MLP F1 Score: {f1_mlp:.4f}')