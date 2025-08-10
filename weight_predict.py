import joblib
knn_model=joblib.load("knn_model.pkl")
result=knn_model.predict([[30,0,1,160]])
print("Result=",result)