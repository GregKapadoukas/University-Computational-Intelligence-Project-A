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
from tensorflow.keras import regularizers

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

# Convert categorical values in the inputs to arithmetical values

sensor_measurements['User'] = sensor_measurements['User'].map({'debora':1, 'katia':2, 'wallace':3, 'jose_carlos':4})
sensor_measurements['Gender'] = sensor_measurements['Gender'].map({'Man':1, 'Woman':2})

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

num_hidden_neurons = 23
num_epochs = 100

# +
preprocessed_df = pd.concat([sensor_measurements, sensor_classes], axis=1)

kf = KFold(n_splits = 5, shuffle = True, random_state = 101)
#result = next(kf.split(preprocessed_df))
num_fold = 0

model_fit_history = []
model_evaluate_history = []


for result in kf.split(preprocessed_df):
    model = keras.Sequential(
        [
            keras.Input(shape=(18)),
            layers.Dense(num_hidden_neurons, activation='relu', kernel_regularizer=regularizers.L2(l2=0.1)),
            layers.Dense(5, activation='softmax')
        ]
    )
    model.summary()

    # Choose loss function, optimizers, learning rate and metrics

    model.compile(
        loss = keras.losses.CategoricalCrossentropy(),
        optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.6, beta_2=0.999),
        metrics = ["categorical_crossentropy", "mse", "categorical_accuracy"]
    )

    # -

    # Split data again into measurements and classes

    # +
    train = preprocessed_df.iloc[result[0]]
    validation =  preprocessed_df.iloc[result[1]]

    train_measurements = train.iloc[:,:18]
    train_classes = train.iloc[:,18:]

    validation_measurements = validation.iloc[:,:18]
    validation_classes = validation.iloc[:,18:]
    # -

    # Create neural network model with keras sequencial API
        
    #def CategoricalAccuracy(y_true, y_pred):
    #    return kb.cast(kb.equal(kb.argmax(y_true, axis=-1), kb.argmax(y_pred, axis=-1)), kb.floatx())

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss')
    #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta = 0.01)
    print("Fold " + str(num_fold+1))
    model_fit_history.append(model.fit(train_measurements, train_classes, batch_size=32, epochs=num_epochs, callbacks=[callback], validation_data=(validation_measurements, validation_classes), verbose=2))

    model_evaluate_history.append(model.evaluate(validation_measurements, validation_classes, batch_size=32, verbose=2))


    num_fold = num_fold + 1

# +
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
#Compare CCE for each epoch
#Fold 1
"""
plt.plot(model_fit_history[0].history['categorical_crossentropy'])
plt.plot(model_fit_history[0].history['val_categorical_crossentropy'])
plt.title('Fold 1 - CCE Loss with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('CCE Loss')
plt.xlabel('Epoch')
x = np.arange(0, len(model_fit_history[0].history['loss']), 1)
#plt.xticks(x + 1)
plt.xlim(1,len(model_fit_history[0].history['loss']))
plt.legend(['Train Set CCE for Fold 1',
            'Validation Set CCE for Fold 1'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

#Fold 2
plt.plot(model_fit_history[1].history['categorical_crossentropy'])
plt.plot(model_fit_history[1].history['val_categorical_crossentropy'])
plt.title('Fold 2 - CCE Loss with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('CCE Loss')
plt.xlabel('Epoch')
x = np.arange(0, len(model_fit_history[1].history['loss']), 1)
#plt.xticks(x + 1)
plt.xlim(1,len(model_fit_history[1].history['loss']))
plt.legend(['Train Set CCE for Fold 2',
            'Validation Set CCE for Fold 2'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

#Fold 3
plt.plot(model_fit_history[2].history['categorical_crossentropy'])
plt.plot(model_fit_history[2].history['val_categorical_crossentropy'])
plt.title('Fold 3 - CCE Loss with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('CCE Loss')
plt.xlabel('Epoch')
x = np.arange(0, len(model_fit_history[2].history['loss']), 1)
#plt.xticks(x + 1)
plt.xlim(1,len(model_fit_history[2].history['loss']))
plt.legend(['Train Set CCE for Fold 3',
            'Validation Set CCE for Fold 3'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

#Fold 4
plt.plot(model_fit_history[3].history['categorical_crossentropy'])
plt.plot(model_fit_history[3].history['val_categorical_crossentropy'])
plt.title('Fold 4 - CCE Loss with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('CCE Loss')
plt.xlabel('Epoch')
x = np.arange(0, len(model_fit_history[3].history['loss']), 1)
#plt.xticks(x + 1)
plt.xlim(1,len(model_fit_history[3].history['loss']))
plt.legend(['Train Set CCE for Fold 4',
            'Validation Set CCE for Fold 4'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

#Fold 5
plt.plot(model_fit_history[4].history['categorical_crossentropy'])
plt.plot(model_fit_history[4].history['val_categorical_crossentropy'])
plt.title('Fold 5 - CCE Loss with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('CCE Loss')
plt.xlabel('Epoch')
x = np.arange(0, len(model_fit_history[4].history['loss']), 1)
#plt.xticks(x + 1)
plt.xlim(1,len(model_fit_history[4].history['loss']))
plt.legend(['Train Set CCE for Fold 5',
            'Validation Set CCE for Fold 5'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()
"""
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
#plt.xticks(x + 1)
plt.xlim(1, max(fold_epochs))
plt.legend(['Validation Set CCE for Fold 1',
            'Validation Set CCE for Fold 2',
            'Validation Set CCE for Fold 3',
            'Validation Set CCE for Fold 4',
            'Validation Set CCE for Fold 5'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

#Compare MSE for each epoch
plt.plot(model_fit_history[0].history['val_mse'])
plt.plot(model_fit_history[1].history['val_mse'])
plt.plot(model_fit_history[2].history['val_mse'])
plt.plot(model_fit_history[3].history['val_mse'])
plt.plot(model_fit_history[4].history['val_mse'])
plt.title('MSE Loss with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
x = np.arange(0, max(fold_epochs), 1)
#plt.xticks(x + 1)
plt.xlim(1, max(fold_epochs))
plt.legend(['Validation Set MSE for Fold 1',
            'Validation Set MSE for Fold 2',
            'Validation Set MSE for Fold 3',
            'Validation Set MSE for Fold 4',
            'Validation Set MSE for Fold 5'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

#Compare Accuracy for each epoch
plt.plot(model_fit_history[0].history['val_categorical_accuracy'])
plt.plot(model_fit_history[1].history['val_categorical_accuracy'])
plt.plot(model_fit_history[2].history['val_categorical_accuracy'])
plt.plot(model_fit_history[3].history['val_categorical_accuracy'])
plt.plot(model_fit_history[4].history['val_categorical_accuracy'])
plt.title('Accuracy with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
x = np.arange(0, max(fold_epochs), 1)
#plt.xticks(x + 1)
plt.xlim(1, max(fold_epochs))
plt.legend(['Validation Set Accuracy Fold 1',
            'Validation Set Accuracy Fold 2',
            'Validation Set Accuracy Fold 3',
            'Validation Set Accuracy Fold 4',
            'Validation Set Accuracy Fold 5'
            ], loc='best')
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

"""
#Plot mean fold CCE
plt.plot(np.mean((model_fit_history[0].history['categorical_crossentropy'],
                model_fit_history[1].history['categorical_crossentropy'],
                model_fit_history[2].history['categorical_crossentropy'],
                model_fit_history[3].history['categorical_crossentropy'],
                model_fit_history[4].history['categorical_crossentropy']), axis=0))
plt.plot(np.mean((model_fit_history[0].history['val_categorical_crossentropy'],
                model_fit_history[1].history['val_categorical_crossentropy'],
                model_fit_history[2].history['val_categorical_crossentropy'],
                model_fit_history[3].history['val_categorical_crossentropy'],
                model_fit_history[4].history['val_categorical_crossentropy']), axis=0))
plt.title('Mean Fold CCE Loss with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('CCE Loss')
plt.xlabel('Epoch')
x = np.arange(0, num_epochs, 1)
#plt.xticks(x + 1)
plt.xlim(1,num_epochs)
plt.legend(['Train Set Mean Fold CCE Error',
            'Validation Set Mean Fold CCE Error'])
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

#Plot mean fold MSE
plt.plot(np.mean((model_fit_history[0].history['mse'],
                model_fit_history[1].history['mse'],
                model_fit_history[2].history['mse'],
                model_fit_history[3].history['mse'],
                model_fit_history[4].history['mse']), axis=0))
plt.title('Mean Fold MSE Loss with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
x = np.arange(0, num_epochs, 1)
#plt.xticks(x + 1)
plt.xlim(1,num_epochs)
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()

#Plot mean fold Accuracy
plt.plot(np.mean((model_fit_history[0].history['categorical_accuracy'],
                model_fit_history[1].history['categorical_accuracy'],
                model_fit_history[2].history['categorical_accuracy'],
                model_fit_history[3].history['categorical_accuracy'],
                model_fit_history[4].history['categorical_accuracy']), axis=0))
plt.title('Mean Fold Accuracy with ' + str(num_hidden_neurons) + ' Neurons in Hidden Layer')
plt.ylabel('CCE Loss')
plt.xlabel('Epoch')
x = np.arange(0, num_epochs, 1)
#plt.xticks(x + 1)
plt.xlim(1,num_epochs)
plt.xticks(np.arange(len(x)), np.arange(1, len(x)+1))
plt.show()
"""
