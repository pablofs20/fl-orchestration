import pandas as pd
from kafka import KafkaProducer
import json
import time

df = pd.read_csv("../datasets/BTS_1_benign.csv")

column_names = df.columns

df_jsons_list = df.to_json(orient='records', lines=True).splitlines()

producer = KafkaProducer(
    bootstrap_servers=['localhost:29092'],
)
for item in df_jsons_list:
    print("yeee")
    print(item)
    print(type(item))
    producer.send("flows-infoo", value=item.encode('utf-8'))
