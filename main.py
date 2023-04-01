# Import necessary libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Read CSV File With Pandas

df = pd.read_csv(
        "dataset-HAR-PUC-Rio.csv",
        names=["User","Gender","Age","Height","Weight","BMI","x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4","Class"],
        delimiter=';',
        #Η γραμμή 0 εξηγεί τις στήλες και η γραμμή 122078 έχει λάθος
        skiprows=[0,122077],
        decimal=','
)

# Separate inputs and known correct output

sensor_measurements = df.copy()
sensor_classes = sensor_measurements.pop("Class")


# One-hot encode output to use in the model later

sensor_classes = pd.get_dummies(df['Class'])
print(sensor_classes)

# Convert categorical values in the inputs to arithmetical values

sensor_measurements['User'] = sensor_measurements['User'].map({'debora':1, 'katia':2, 'wallace':3, 'jose_carlos':4})
sensor_measurements['Gender'] = sensor_measurements['Gender'].map({'Man':1, 'Woman':2})
print(sensor_measurements)

# Centering of input data

sensor_measurements = sensor_measurements.apply(lambda x: x-x.mean())
print(sensor_measurements)

# Normalization of input data

for column in sensor_measurements.columns:
    sensor_measurements[column] = (sensor_measurements[column] - sensor_measurements[column].min()) / (sensor_measurements[column].max() - sensor_measurements[column].min())    
print(sensor_measurements)

# Standardization of input data

# +
#for column in sensor_measurements.columns:
#    sensor_measurements[column] = (sensor_measurements[column] - sensor_measurements[column].mean()) / (sensor_measurements[column].std())    
#print(sensor_measurements)
# -

# Merge back measurements and classes and split data for CV (5-fold)

# +
preprocessed_df = pd.concat([sensor_measurements, sensor_classes], axis=1)

kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
result = next(kf.split(preprocessed_df), None)

train = preprocessed_df.iloc[result[0]]
test =  preprocessed_df.iloc[result[1]]
# -

# Split data again into measurements and classes

# +
train_measurements = train.iloc[:,:18]
train_classes = train.iloc[:,18:]

test_measurements = test.iloc[:,:18]
test_classes = test.iloc[:,18:]
# -

# Create neural network model with keras sequencial API

model = keras.Sequential(
    [
        keras.Input(shape=(18)),
        layers.Dense(20, activation='relu'),
        layers.Dense(5, activation='softmax')
    ]
)
model.summary()

# Choose loss function, optimizers, learning rate and metrics

model.compile(
    loss = keras.losses.CategoricalCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    metrics = ["accuracy"]
)

# Train the model and evaluate

model.fit(train_measurements, train_classes, batch_size=32, epochs=5, verbose=2)
model.evaluate(test_measurements, test_classes, batch_size=32, verbose=2)
