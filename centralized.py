import pandas as pd
import numpy as np
from preprocessor import preprocess_data
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.losses import mean_squared_error
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def create_basic_autoencoder(X):
    n_features = X.shape[1]

    encoding_neurons = round(n_features/2)
    hidden_neurons_1 = round(encoding_neurons/2)
    hidden_neurons_2 = round(hidden_neurons_1/2)

    learning_rate = 1e-7

    inputs = Input(shape=(n_features,))

    # encoder
    encoder = Dense(encoding_neurons, activation="tanh",
                    activity_regularizer=regularizers.l2(learning_rate))(inputs)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(hidden_neurons_1, activation='relu')(encoder)
    encoder = Dense(hidden_neurons_2, activation="relu")(encoder)

    # decoder
    decoder = Dense(hidden_neurons_1, activation='relu')(encoder)
    decoder = Dropout(0.2)(decoder)
    decoder = Dense(encoding_neurons, activation='relu')(decoder)
    decoder = Dense(n_features, activation='tanh')(decoder)

    autoencoder = Model(inputs=inputs, outputs=decoder)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


df = pd.read_csv("BTS_1.csv")
df = df.drop(columns=["Unnamed: 0"])
df["Label"] = df['Label'].replace('Benign', 0).replace('Malicious', 1)
benign_df = df.loc[df["Label"] == 0]
benign_df = benign_df.sample(n=100000)
malign_df = df.loc[df["Label"] == 1]
malign_df.to_csv("malign.csv", index=False)
malign_df = malign_df.sample(n=100000)

benign_df.drop(columns=["Label"], inplace=True)
malign_df.drop(columns=["Label"], inplace=True)

X_train, X_test = preprocess_data(benign_df, 0.3)


autoencoder = create_basic_autoencoder(X_train)

autoencoder.fit(X_train, X_train, epochs=10)
