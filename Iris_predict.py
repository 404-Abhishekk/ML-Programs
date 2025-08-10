import joblib as jb
knn_iris_model=jb.load("knn_iris_model.pkl")
result=knn_iris_model.predict([[6.1,3.7,4.4,1]])
print(result)