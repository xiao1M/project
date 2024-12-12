import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

#获取数据
data = pd.read_csv('data.csv')

# 第一列为id，最后一列是target，其他列是特征
x = data.iloc[:, 1:-1].values  # 特征数据
y = data.iloc[:, -1].values   # 目标数据


#划分数据集
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#KNN算法预估器
estimator = KNeighborsClassifier(n_neighbors=1)
estimator.fit(x_train,y_train)

#模型评估
y_predict = estimator.predict(x_test)
print("y_predict:\n",y_predict)
print("比对真实值和预测值:\n",y_test == y_predict)

print("准确率为:\n",estimator.score(x_test, y_test))
print("f1-score为:\n",f1_score(y_test,y_predict,average = 'macro'))