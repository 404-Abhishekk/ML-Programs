import joblib as jb
import numpy as np
model = jb.load('redwine_model.pkl')
result = model.predict(np.array([[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]))
print("Predicted Quality=", result)