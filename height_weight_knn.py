import pandas as pd 
import sklearn.neighbors as ng 
import matplotlib.pyplot as mt 

mydata=pd.read_csv("data.csv")
x=mydata[["height"]]
y=mydata[["weight"]]
knn_model=ng.KNeighborsRegressor(n_neighbors=3)
knn_model.fit(x,y)
print(knn_model.predict([[160]]))