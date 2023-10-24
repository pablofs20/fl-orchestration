from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras import regularizers

def create_basic_autoencoder(n_features):
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
