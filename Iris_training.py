import pandas as pd
import sklearn.neighbors as ng
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import joblib as jb

mydata=pd.read_csv('Iris.csv')
x=mydata[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=mydata[['Species']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
knn_iris_model=ng.KNeighborsClassifier(n_neighbors=5)
knn_iris_model.fit(x,y)
jb.dump(knn_iris_model,'knn_iris_model.pkl')
print("Training complete")
test_result=knn_iris_model.predict(x_test)
print("Accuracy score:",round(accuracy_score(y_test,test_result),2)*100)
print("Confusion matrix",confusion_matrix(y_test,test_result))