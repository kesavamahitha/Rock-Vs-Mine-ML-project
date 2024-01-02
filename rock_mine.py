import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
filepath = r"sonar data.csv"
data = pd.read_csv(filepath, header = None)
X = data.drop(columns = 60, axis = 1)
Y = data[60]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1) 
model = LogisticRegression()
model.fit(X_train, Y_train)
#training data prediction
train_data_prediction = model.predict(X_train)
accuracy_of_training_data = accuracy_score(train_data_prediction,  Y_train)
print(accuracy_of_training_data)
#testing data prediction
test_data_prediction = model.predict(X_test)
accuracy_of_testing_data = accuracy_score(test_data_prediction,  Y_test)
print(accuracy_of_testing_data)
input_data = (0.0762,0.0666,0.0481,0.0394,0.059,0.0649,0.1209,0.2467,0.3564,0.4459,0.4152,0.3952,0.4256,0.4135,0.4528,0.5326,0.7306,0.6193,0.2032,0.4636,0.4148,0.4292,0.573,0.5399,0.3161,0.2285,0.6995,1,0.7262,0.4724,0.5103,0.5459,0.2881,0.0981,0.1951,0.4181,0.4604,0.3217,0.2828,0.243,0.1979,0.2444,0.1847,0.0841,0.0692,0.0528,0.0357,0.0085,0.023,0.0046,0.0156,0.0031,0.0054,0.0105,0.011,0.0015,0.0072,0.0048,0.0107,0.0094
)
input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)
prediction = model.predict(input_data)
print(prediction)
input_data = (0.1313,0.2339,0.3059,0.4264,0.401,0.1791,0.1853,0.0055,0.1929,0.2231,0.2907,0.2259,0.3136,0.3302,0.366,0.3956,0.4386,0.467,0.5255,0.3735,0.2243,0.1973,0.4337,0.6532,0.507,0.2796,0.4163,0.595,0.5242,0.4178,0.3714,0.2375,0.0863,0.1437,0.2896,0.4577,0.3725,0.3372,0.3803,0.4181,0.3603,0.2711,0.1653,0.1951,0.2811,0.2246,0.1921,0.15,0.0665,0.0193,0.0156,0.0362,0.021,0.0154,0.018,0.0013,0.0106,0.0127,0.0178,0.0231)
input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)
prediction = model.predict(input_data)
print(prediction)
