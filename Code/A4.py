#Import necessary libraries

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
from tensorflow.keras import regularizers

#Read CSV File With Pandas
df = pd.read_csv(
        "dataset-HAR-PUC-Rio.csv",
        names=["User","Gender","Age","Height","Weight","BMI","x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4","Class"],
        delimiter=';',
        #Line 0 contains information about the columns and line 122078 has a mistake
        skiprows=[0,122077],
        decimal=','
)

#Separate inputs and known correct output
sensor_measurements = df.copy()
sensor_classes = sensor_measurements.pop("Class")

#One-hot encode output to use in the model later
sensor_classes = pd.get_dummies(df['Class'])

#Convert categorical values in the inputs to arithmetical values
sensor_measurements['User'] = sensor_measurements['User'].map({'debora':1, 'katia':2, 'wallace':3, 'jose_carlos':4})
sensor_measurements['Gender'] = sensor_measurements['Gender'].map({'Man':1, 'Woman':2})

#Centering of input data
#sensor_measurements = sensor_measurements.apply(lambda x: x-x.mean())

#Normalization of input data
#scaler = MinMaxScaler(feature_range=(-1, 1))
#sensor_measurements = pd.DataFrame(scaler.fit_transform(sensor_measurements), columns=sensor_measurements.columns)

#Standardization of input data
for column in sensor_measurements.columns:
    sensor_measurements[column] = (sensor_measurements[column] - sensor_measurements[column].mean()) / (sensor_measurements[column].std())    

#Set number of neurons in hidden layer and number of max epochs
num_hidden_neurons = 23
num_epochs = 100

#Join measurements and classes again in order to split for 5-fold CV
preprocessed_df = pd.concat([sensor_measurements, sensor_classes], axis=1)
kf = KFold(n_splits = 5, shuffle = True, random_state = 101)

#Store fold number
num_fold = 0

#Store fit and evaluation results for each fold
model_fit_history = []
model_evaluate_history = []

#Run once for each fold
for result in kf.split(preprocessed_df):

    #Define model
    model = keras.Sequential(
        [
            keras.Input(shape=(18)),
            layers.Dense(num_hidden_neurons, activation='relu', kernel_regularizer=regularizers.L2(l2=0.1)),
            layers.Dense(5, activation='softmax')
        ]
    )
    model.summary()

    #Choose loss function, optimizers, learning rate, beta values and metrics
    model.compile(
        loss = keras.losses.CategoricalCrossentropy(),
        optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.6, beta_2=0.999),
        metrics = ["categorical_crossentropy", "mse", "categorical_accuracy"]
    )

    #Split data again into measurements and classes
    train = preprocessed_df.iloc[result[0]]
    validation =  preprocessed_df.iloc[result[1]]

    train_measurements = train.iloc[:,:18]
    train_classes = train.iloc[:,18:]

    validation_measurements = validation.iloc[:,:18]
    validation_classes = validation.iloc[:,18:]

    #Define callback for early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss')

    #Train the model and evaluate
    print("Fold " + str(num_fold+1))
    model_fit_history.append(model.fit(train_measurements, train_classes, batch_size=32, epochs=num_epochs, callbacks=[callback], validation_data=(validation_measurements, validation_classes), verbose=2))

    model_evaluate_history.append(model.evaluate(validation_measurements, validation_classes, batch_size=32, verbose=2))

    #Add one to fold number for next iteration
    num_fold = num_fold + 1

#Calculate and print mean values for evaluations for all folds
model_categorical_crossentropy = 0
model_mse = 0
model_categorical_accuracy = 0

for i in range(0,5):
    model_categorical_crossentropy = model_categorical_crossentropy + model_evaluate_history[i][1]
    model_mse = model_mse + model_evaluate_history[i][2]
    model_categorical_accuracy = model_categorical_accuracy + model_evaluate_history[i][3]

model_categorical_crossentropy = model_categorical_crossentropy / 5
model_mse = model_mse / 5
model_categorical_accuracy = model_categorical_accuracy / 5
print("Mean Evaluation Result From Evaluation Sets in 5-fold CV")
print("Categorical CE: " + str(model_categorical_crossentropy) + ", MSE: " + str(model_mse) + ", Accuracy: " + str(model_categorical_accuracy))

#Plot model results
#Compare validation set metrics for each fold
#Fold 1
#Compare CCE
fold_epochs = [len(model_fit_history[0].history['loss']), 
               len(model_fit_history[1].history['loss']), 
               len(model_fit_history[2].history['loss']), 
               len(model_fit_history[3].history['loss']), 
               len(model_fit_history[4].history['loss'])
               ]
plt.plot(model_fit_history[0].history['val_categorical_crossentropy'])
plt.plot(model_fit_history[1].history['val_categorical_crossentropy'])
plt.plot(model_fit_history[2].history['val_categorical_crossentropy'])
plt.plot(model_fit_history[3].history['val_categorical_crossentropy'])
plt.plot(model_fit_history[4].history['val_categorical_crossentropy'])
plt.title('CCE Loss with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('CCE Loss')
plt.xlabel('Epoch')
x = np.arange(0, max(fold_epochs), 1)
plt.xlim(1, max(fold_epochs))
plt.legend(['Validation Set CCE for Fold 1',
            'Validation Set CCE for Fold 2',
            'Validation Set CCE for Fold 3',
            'Validation Set CCE for Fold 4',
            'Validation Set CCE for Fold 5'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

#Compare MSE
plt.plot(model_fit_history[0].history['val_mse'])
plt.plot(model_fit_history[1].history['val_mse'])
plt.plot(model_fit_history[2].history['val_mse'])
plt.plot(model_fit_history[3].history['val_mse'])
plt.plot(model_fit_history[4].history['val_mse'])
plt.title('MSE Loss with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
x = np.arange(0, max(fold_epochs), 1)
plt.xlim(1, max(fold_epochs))
plt.legend(['Validation Set MSE for Fold 1',
            'Validation Set MSE for Fold 2',
            'Validation Set MSE for Fold 3',
            'Validation Set MSE for Fold 4',
            'Validation Set MSE for Fold 5'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

#Compare Accuracy
plt.plot(model_fit_history[0].history['val_categorical_accuracy'])
plt.plot(model_fit_history[1].history['val_categorical_accuracy'])
plt.plot(model_fit_history[2].history['val_categorical_accuracy'])
plt.plot(model_fit_history[3].history['val_categorical_accuracy'])
plt.plot(model_fit_history[4].history['val_categorical_accuracy'])
plt.title('Accuracy with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
x = np.arange(0, max(fold_epochs), 1)
plt.xlim(1, max(fold_epochs))
plt.legend(['Validation Set Accuracy Fold 1',
            'Validation Set Accuracy Fold 2',
            'Validation Set Accuracy Fold 3',
            'Validation Set Accuracy Fold 4',
            'Validation Set Accuracy Fold 5'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()
