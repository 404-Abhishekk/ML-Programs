import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error
import sklearn.neighbors as ng
import joblib
import math 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

model=Sequential()
model.add(Dense(10,activation="relu",input_shape=(4,)))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(X_train,Y_train,epochs=100)