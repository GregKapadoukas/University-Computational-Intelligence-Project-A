# Import necessary libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import keras.backend as kb

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
sensor_classes

# Convert categorical values in the inputs to arithmetical values

sensor_measurements['User'] = sensor_measurements['User'].map({'debora':1, 'katia':2, 'wallace':3, 'jose_carlos':4})
sensor_measurements['Gender'] = sensor_measurements['Gender'].map({'Man':1, 'Woman':2})
sensor_measurements

# Centering of input data

# +
#sensor_measurements = sensor_measurements.apply(lambda x: x-x.mean())
#print(sensor_measurements)
# -

# Normalization of input data

# +
#scaler = MinMaxScaler(feature_range=(-1, 1))
#sensor_measurements = pd.DataFrame(scaler.fit_transform(sensor_measurements), columns=sensor_measurements.columns)
# -

# Standardization of input data

for column in sensor_measurements.columns:
    sensor_measurements[column] = (sensor_measurements[column] - sensor_measurements[column].mean()) / (sensor_measurements[column].std())    
sensor_measurements

num_hidden_neurons = 23
num_epochs = 10

# +
preprocessed_df = pd.concat([sensor_measurements, sensor_classes], axis=1)

kf = KFold(n_splits = 5, shuffle = True, random_state = 101)
#result = next(kf.split(preprocessed_df))
num_fold = 0

model1_fit_history = []
model2_fit_history = []
model1_evaluate_history = []
model2_evaluate_history = []

for result in kf.split(preprocessed_df):
    model1 = keras.Sequential(
        [
            keras.Input(shape=(18)),
            layers.Dense(num_hidden_neurons, activation='relu'),
            layers.Dense(5, activation='softmax')
        ]
    )
    model1.summary()

    # Choose loss function, optimizers, learning rate and metrics

    model1.compile(
        loss = keras.losses.CategoricalCrossentropy(),
        optimizer = keras.optimizers.Adam(learning_rate = 0.001),
        metrics = ["categorical_crossentropy", "mse", "categorical_accuracy"]
    )

    model2 = keras.Sequential(
        [
            keras.Input(shape=(18)),
            layers.Dense(num_hidden_neurons, activation='relu'),
            layers.Dense(5, activation='softmax')
        ]
    )

    model2.summary()

    # Choose loss function, optimizers, learning rate and metrics

    model2.compile(
        loss = keras.losses.MeanSquaredError(),
        optimizer = keras.optimizers.Adam(learning_rate = 0.001),
        metrics = ["categorical_crossentropy", "mse", "categorical_accuracy"]
    )

    # -

    # Split data again into measurements and classes

    # +
    train = preprocessed_df.iloc[result[0]]
    test =  preprocessed_df.iloc[result[1]]

    train_measurements = train.iloc[:,:18]
    train_classes = train.iloc[:,18:]

    test_measurements = test.iloc[:,:18]
    test_classes = test.iloc[:,18:]
    # -

    # Create neural network model with keras sequencial API
        
    #def CategoricalAccuracy(y_true, y_pred):
    #    return kb.cast(kb.equal(kb.argmax(y_true, axis=-1), kb.argmax(y_pred, axis=-1)), kb.floatx())

    print("Fold " + str(num_fold+1))
    print("Model trained with CCE")
    model1_fit_history.append(model1.fit(train_measurements, train_classes, batch_size=32, epochs=num_epochs, verbose=2))
    print("Model trained with MSE")
    model2_fit_history.append(model2.fit(train_measurements, train_classes, batch_size=32, epochs=num_epochs, verbose=2))

    print("CCE Trained Model")
    model1_evaluate_history.append(model1.evaluate(test_measurements, test_classes, batch_size=32, verbose=2))

    print("MSE Trained Model")
    model2_evaluate_history.append(model2.evaluate(test_measurements, test_classes, batch_size=32, verbose=2))

    num_fold = num_fold + 1
model1_evaluate_history

# +
model1_categorical_crossentropy = 0
model1_mse = 0
model1_categorical_accuracy = 0
model2_categorical_crossentropy = 0
model2_mse = 0
model2_categorical_accuracy = 0

for i in range(0,5):
    model1_categorical_crossentropy = model1_categorical_crossentropy + model1_evaluate_history[i][1]
    model1_mse = model1_mse + model1_evaluate_history[i][2]
    model1_categorical_accuracy = model1_categorical_accuracy + model1_evaluate_history[i][3]
    model2_categorical_crossentropy = model2_categorical_crossentropy + model2_evaluate_history[i][1]
    model2_mse = model1_mse + model2_evaluate_history[i][2]
    model2_categorical_accuracy = model2_categorical_accuracy + model2_evaluate_history[i][3]

model1_categorical_crossentropy = model1_categorical_crossentropy / 5
model1_mse = model1_mse / 5
model1_categorical_accuracy = model1_categorical_accuracy / 5
model2_categorical_crossentropy = model2_categorical_crossentropy / 5
model2_mse = model2_mse / 5
model2_categorical_accuracy = model2_categorical_accuracy / 5
print("Mean Evaluation Result From Test Sets in 5-fold CV")
print("Model 1: Categorical CE: " + str(model1_categorical_crossentropy) + ", MSE: " + str(model1_mse) + ", Accuracy: " + str(model1_categorical_accuracy))
print("Model 2: Categorical CE: " + str(model2_categorical_crossentropy) + ", MSE: " + str(model2_mse) + ", Accuracy: " + str(model2_categorical_accuracy))

#Plot model results
#Compare CCE for CCE Trained and MSE Trained
plt.plot(model1_fit_history[0].history['categorical_crossentropy'])
plt.plot(model1_fit_history[1].history['categorical_crossentropy'])
plt.plot(model1_fit_history[2].history['categorical_crossentropy'])
plt.plot(model1_fit_history[3].history['categorical_crossentropy'])
plt.plot(model1_fit_history[4].history['categorical_crossentropy'])
plt.plot(model2_fit_history[0].history['categorical_crossentropy'])
plt.plot(model2_fit_history[1].history['categorical_crossentropy'])
plt.plot(model2_fit_history[2].history['categorical_crossentropy'])
plt.plot(model2_fit_history[3].history['categorical_crossentropy'])
plt.plot(model2_fit_history[4].history['categorical_crossentropy'])
plt.title('Categorical CE vs MSE Trained NN - CCE Loss with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('CCE Loss')
plt.xlabel('Epoch')
x = np.arange(0, num_epochs, 1)
#plt.xticks(x + 1)
plt.xlim(1,num_epochs)
plt.legend(['CCE for CCE Trained NN Fold 1',
            'CCE for CCE Trained NN Fold 2',
            'CCE for CCE Trained NN Fold 3',
            'CCE for CCE Trained NN Fold 4',
            'CCE for CCE Trained NN Fold 5',
            'CCE for MSE Trained NN Fold 1',
            'CCE for MSE Trained NN Fold 2',
            'CCE for MSE Trained NN Fold 3',
            'CCE for MSE Trained NN Fold 4',
            'CCE for MSE Trained NN Fold 5'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

#Compare MSE for CCE Trained and MSE Trained
plt.plot(model1_fit_history[0].history['mse'])
plt.plot(model1_fit_history[1].history['mse'])
plt.plot(model1_fit_history[2].history['mse'])
plt.plot(model1_fit_history[3].history['mse'])
plt.plot(model1_fit_history[4].history['mse'])
plt.plot(model2_fit_history[0].history['mse'])
plt.plot(model2_fit_history[1].history['mse'])
plt.plot(model2_fit_history[2].history['mse'])
plt.plot(model2_fit_history[3].history['mse'])
plt.plot(model2_fit_history[4].history['mse'])
plt.title('Categorical CE vs MSE Trained NN - MSE Loss with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
x = np.arange(0, num_epochs, 1)
#plt.xticks(x + 1)
plt.xlim(1,num_epochs)
plt.legend(['MSE for CCE Trained NN Fold 1',
            'MSE for CCE Trained NN Fold 2',
            'MSE for CCE Trained NN Fold 3',
            'MSE for CCE Trained NN Fold 4',
            'MSE for CCE Trained NN Fold 5',
            'MSE for MSE Trained NN Fold 1',
            'MSE for MSE Trained NN Fold 2',
            'MSE for MSE Trained NN Fold 3',
            'MSE for MSE Trained NN Fold 4',
            'MSE for MSE Trained NN Fold 5'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

#Compare Accuracy for CCE Trained and MSE Trained
plt.plot(model1_fit_history[0].history['categorical_accuracy'])
plt.plot(model1_fit_history[1].history['categorical_accuracy'])
plt.plot(model1_fit_history[2].history['categorical_accuracy'])
plt.plot(model1_fit_history[3].history['categorical_accuracy'])
plt.plot(model1_fit_history[4].history['categorical_accuracy'])
plt.plot(model2_fit_history[0].history['categorical_accuracy'])
plt.plot(model2_fit_history[1].history['categorical_accuracy'])
plt.plot(model2_fit_history[2].history['categorical_accuracy'])
plt.plot(model2_fit_history[3].history['categorical_accuracy'])
plt.plot(model2_fit_history[4].history['categorical_accuracy'])
plt.title('Categorical CE vs MSE Trained NN - Accuracy with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
x = np.arange(0, num_epochs, 1)
#plt.xticks(x + 1)
plt.xlim(1,num_epochs)
plt.legend(['Accuracy for CCE Trained NN Fold 1',
            'Accuracy for CCE Trained NN Fold 2',
            'Accuracy for CCE Trained NN Fold 3',
            'Accuracy for CCE Trained NN Fold 4',
            'Accuracy for CCE Trained NN Fold 5',
            'Accuracy for MSE Trained NN Fold 1',
            'Accuracy for MSE Trained NN Fold 2',
            'Accuracy for MSE Trained NN Fold 3',
            'Accuracy for MSE Trained NN Fold 4',
            'Accuracy for MSE Trained NN Fold 5'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()
