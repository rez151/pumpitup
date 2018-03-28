import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from keras import backend as K

# best loss / accuracy
best_loss = 1
best_loss_setup = ""
best_accuracy = 0
best_accuracy_setup = ""

# network and training
FEATURES = 4
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
VERBOSE = 1

LAYER = 3
N_HIDDEN = 30
DROPOUT = 0.0
EPOCHS = 500

# Load Training Data
training_data_df = pd.read_csv("watertraining.csv")
training_label_data_df = pd.read_csv("traininglabels.csv")

X = training_data_df.drop('id', axis=1).values
Y = training_label_data_df[['status_group']].values

# Load the separate test data set
test_data_df = pd.read_csv("watertest.csv")

X_test = test_data_df.drop('id', axis=1).values

# K.clear_session()

# Define the model
model = Sequential()

if LAYER == 1:
    model.add(Dense(1, input_dim=FEATURES, kernel_initializer='uniform', activation='softmax'))

else:
    model.add(Dense(N_HIDDEN, input_dim=FEATURES, kernel_initializer='uniform', activation='relu'))
    current_layers = 2

    while current_layers < LAYER:
        model.add(Dense(N_HIDDEN, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(DROPOUT))
        current_layers += 1

    model.add(Dense(1, kernel_initializer='uniform', activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

RUN_NAME = "La" + str(LAYER) + "No" + str(N_HIDDEN) + "Dr" + str(DROPOUT) + "Ep" + str(EPOCHS)

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='logs/' + RUN_NAME,
    histogram_freq=5,
    write_graph=True
)

# Train the model
model.fit(
    X,
    Y,
    epochs=EPOCHS,
    shuffle=True,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[logger]

)

setup = "Epochs: " + str(EPOCHS) + "  Layer: " + str(LAYER) + "   Nodes: " + str(
    N_HIDDEN) + "    Dropout: " + str(
    DROPOUT)
score = model.evaluate(X_test, Y_test, verbose=0)
print("---------------------------------")
print(setup)
print("Test score/loss: ", score[0])
print("Test accuracy: ", score[1])
