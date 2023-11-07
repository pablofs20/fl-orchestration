import pandas as pd
from keras.models import load_model
from keras.losses import mean_squared_error
from preprocessor import label_encode

# Cargamos modelo y threshold de detección (MSE)
model = load_model("model_centralized.h5")

anomaly_threshold = 2.80202

# Cargamos datasets de prueba (benignos y malignos)
bs1_benign = pd.read_csv("../datasets/BTS_1_benign.csv").sample(n=5000)
bs2_benign = pd.read_csv("../datasets/BTS_2_benign.csv").sample(n=5000)

bs1_malign = pd.read_csv("../datasets/BTS_1_malign.csv").sample(n=5000)
bs2_malign = pd.read_csv("../datasets/BTS_2_malign.csv").sample(n=5000)

benign_df = label_encode(pd.concat([bs1_benign, bs2_benign])).to_numpy()
malign_df = label_encode(pd.concat([bs1_malign, bs2_malign])).to_numpy()

# Hacemos el predict de cada uno de los 2 datasets de prueba
print(benign_df)
benign_pred = model.predict(benign_df)
malign_pred = model.predict(malign_df)

# Sacamos el MSE de cada uno
benign_mse = mean_squared_error(benign_df, benign_pred).numpy()
malign_mse = mean_squared_error(malign_df, malign_pred).numpy()

print(benign_mse)
print(malign_mse)
