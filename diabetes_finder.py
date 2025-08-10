import pandas as pd
import sklearn.neighbors as ng

mydata=pd.read_csv("diabetes.csv")

x=mydata[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y=mydata[["Outcome"]]

model=ng.KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)
result=model.predict([[0,80,72,35,0,33.6,0.627,50]])
if result[0]==1:
    print("Diabetic person")
else:
    print("Not a diabetic person")