import pandas as pd
import numpy as np
from keras.losses import mean_squared_error
from preprocessor import preprocess_data_centralized
from model_creator import create_basic_autoencoder

BS1_benign = pd.read_csv("../datasets/BTS_1_benign.csv").sample(n=50000)
BS2_benign = pd.read_csv("../datasets/BTS_2_benign.csv").sample(n=50000)

df = pd.concat([BS1_benign, BS2_benign])

X_train, X_test, n_features = preprocess_data_centralized(df, 0.3)

autoencoder = create_basic_autoencoder(n_features)

autoencoder.fit(X_train, X_train, epochs=50)

predictions = autoencoder.predict(X_train)
train_mse = mean_squared_error(X_train, predictions).numpy()

print(train_mse)
print(type(train_mse))
threshold = np.percentile(train_mse, 95)
print(threshold)

autoencoder.save("model_centralized.h5")
