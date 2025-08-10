import joblib
knn_model=joblib.load("knn_model.pkl")
result=knn_model.predict([[2,1,0,125,212,1,168,0,1,2,2,3]])
print(result)