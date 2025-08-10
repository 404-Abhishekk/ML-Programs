import pandas as pd
import matplotlib.pyplot as mt
import sklearn.linear_model as sk
mydata=pd.read_csv("studyhrs.csv")
x=mydata[["studyhrs"]]
y=mydata[["score"]]
mt.scatter(x,y)
mt.show()
model=sk.LinearRegression()
model.fit(x,y)
print(model.predict([[7]]))