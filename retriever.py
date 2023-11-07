import requests
import tensorflow as tf
import pandas as pd
from keras.losses import mean_squared_error
from preprocessor import preprocess_data_client_side

url = 'http://localhost:7777/getModel'  # Ajusta la URL según la dirección de tu servidor Flask
response = requests.get(url)

if response.status_code == 200:
    model_data = response.json()
    # Aquí puedes cargar el modelo desde 'model_data' y utilizarlo en tu aplicación cliente.
else:
    print("Error al obtener el modelo.")

print(model_data)

import tensorflow as tf

# Suponiendo que 'model_data' es un diccionario JSON que contiene el modelo serializado
model_json = model_data['model']

# Carga el modelo desde JSON
model = tf.keras.models.model_from_json(model_json)

# Ahora puedes utilizar 'loaded_model' en tu aplicación cliente
print(model)

anomaly_threshold = 2.80202

# Cargamos datasets de prueba (benignos y malignos)
bs1_benign = pd.read_csv("../datasets/BTS_1_benign.csv").sample(n=5000)
bs2_benign = pd.read_csv("../datasets/BTS_2_benign.csv").sample(n=5000)

bs1_malign = pd.read_csv("../datasets/BTS_1_malign.csv").sample(n=5000)
bs2_malign = pd.read_csv("../datasets/BTS_2_malign.csv").sample(n=5000)

benign_df = pd.concat([bs1_benign, bs2_benign])
malign_df = pd.concat([bs1_malign, bs2_malign])

aggregator_api_uri = "http://localhost:9090/getVariables"
response = requests.get(aggregator_api_uri)
aggregator_selected_columns = list(response.json())

og_columns = list(benign_df.columns)
difference = list(set(og_columns) - set(aggregator_selected_columns))
benign_df = benign_df.drop(columns=difference, axis=1)
malign_df = malign_df.drop(columns=difference, axis=1)


benign_df_X_train, benign_df_X_test, cols = preprocess_data_client_side(benign_df, 0.33)
malign_df_X_train, malign_df_X_test, cols = preprocess_data_client_side(malign_df, 0.33)

# Hacemos el predict de cada uno de los 2 datasets de prueba
benign_pred = model.predict(benign_df_X_test)
malign_pred = model.predict(malign_df_X_test)

# Sacamos el MSE de cada uno
benign_mse = mean_squared_error(benign_df_X_test, benign_pred).numpy()
malign_mse = mean_squared_error(malign_df_X_test, malign_pred).numpy()

print(benign_mse)
print(malign_mse)

