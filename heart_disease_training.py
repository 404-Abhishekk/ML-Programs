import pandas as pd
import sklearn.neighbors as ng 
import joblib 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import math

mydata=pd.read_csv("heart.csv")

x=mydata[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y=mydata[["target"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
knn_model=ng.KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train,y_train)

joblib.dump(knn_model,"knn_model.pkl")
print("Training has completed successfully")

test_result=knn_model.predict(x_test)
print("Accuracy score",accuracy_score(y_test,test_result)*100)