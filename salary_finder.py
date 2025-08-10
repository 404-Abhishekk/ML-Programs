import pandas as pd 
import matplotlib.pyplot as mt
from  sklearn.preprocessing import LabelEncoder

mydata=pd.read_csv("salary_data.csv")
x=mydata[["education_qualification"]]
y=mydata[["salary"]]
le=LabelEncoder()
x_new=le.fit_transform(x)
print(x)
print(x_new)
mt.scatter(x_new,y)
mt.show()