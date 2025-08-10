import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error
import sklearn.neighbors as ng
import joblib
import math 

df=pd.read_csv("dataset_weight_finder.csv")
GenderEncoder=LabelEncoder()
BodyTypeEncoder=LabelEncoder()

df["Gender_enc"]=GenderEncoder.fit_transform(df["Gender"])
df["BodyType_enc"]=BodyTypeEncoder.fit_transform(df["BodyType"])

X=df[["Age","Gender_enc","BodyType_enc","Height"]]
Y=df[["Weight"]]

print(X)
print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
knn_model=ng.KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train,Y_train)

joblib.dump(knn_model,"knn_model.pkl")
print("Training successful")
test_result=knn_model.predict(X_test)
print("MSE=",mean_squared_error(Y_test,test_result))
print("RMSE=",(math.sqrt(mean_squared_error(Y_test,test_result))))
print("R2_score=",r2_score(Y_test,test_result))