import pandas as pd
from FLClient import FLClient
import requests

df = pd.read_csv("../datasets/BTS_1_benign.csv")

df = df.sample(n=50000)

aggregator_api_uri = "http://{aggregator_ip}:{aggregator_api_port}/getVariables".format(aggregator_ip="localhost", aggregator_api_port=9090)
response = requests.get(aggregator_api_uri)
aggregator_selected_columns = list(response.json())

og_columns = list(df.columns)
difference = list(set(og_columns) - set(aggregator_selected_columns))
df = df.drop(columns=difference, axis=1)

client = FLClient(df)

client.start()

model = client.get_final_model()
model.save("model_200R.h5")

print(client.get_detection_threshold)
