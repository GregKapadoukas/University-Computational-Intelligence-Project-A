import os
from pandas.io.parsers.readers import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import int64, float64, keras
from keras import layers
import pandas as pd
import numpy as np

#Reading of CSV file
csv_data = pd.read_csv(
        "dataset-HAR-PUC-Rio.csv",
        names=["User","Gender","Age","Height","Weight","BMI","x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4","Class"],
        delimiter=';',
        #Η γραμμή 0 εξηγεί τις στήλες και η γραμμή 122078 έχει λάθος
        skiprows=[0,122077],
        decimal=','
)

csv_data.head()
csv_data.info()
#print(csv_data)

csv_measurements = csv_data.copy()
csv_classes = csv_measurements.pop("Class")

#Preprocessing Model
inputs = {}
for name, column in csv_measurements.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float64
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float64}
x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(csv_data[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)
preprocessed_inputs = [all_numeric_inputs]


for name, input in inputs.items():
  if input.dtype == tf.float64:
    continue

  lookup = layers.StringLookup(vocabulary=np.unique(csv_measurements[name]))
  one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True, to_file='model.png')
