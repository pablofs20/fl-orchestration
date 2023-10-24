import pandas as pd
from kafka import KafkaProducer
import json
import time

df = pd.read_csv("BTS_1.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)

df["Label"] = df['Label'].replace('Benign', 0).replace('Malicious', 1)
benign_df = df.loc[df["Label"] == 0]
malign_df = df.loc[df["Label"] == 1]

print(df['Label'].value_counts())

benign_df.drop(columns=["Label"], inplace=True)
malign_df.drop(columns=["Label"], inplace=True)

column_names = df.columns

benign_df_jsons_list = benign_df.to_json(orient='records', lines=True).splitlines()
malign_df_jsons_list = malign_df.to_json(orient='records', lines=True).splitlines()

producer = KafkaProducer(
    bootstrap_servers=['localhost:29092'],
)
print(len(malign_df_jsons_list))
for item in benign_df_jsons_list:
    print("yeee")
    print(item)
    print(type(item))
    producer.send("flows-infoo", value=item.encode('utf-8'))
