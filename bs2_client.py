from FLClient import FLClient
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle as pkl

df = pd.read_csv("BTS_1.csv")
df = df.drop(columns=["Unnamed: 0"])

df["Label"] = df['Label'].replace('Benign', 0).replace('Malicious', 1)
benign_df = df.loc[df["Label"] == 0]
malign_df = df.loc[df["Label"] == 1]

benign_df = benign_df.drop(columns=["Label"])
malign_df = malign_df.drop(columns=["Label"])

client = FLClient(benign_df)

client.start()

print(client.get_detection_threshold())