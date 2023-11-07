#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from preprocessor import preprocess_data_client_side
from model_creator import create_basic_autoencoder
from keras.losses import mean_squared_error
import configparser
import flwr as fl
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


CONFIG_FILE = "resources/fl_agent.ini"

def get_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    client_config = config["FL Client"]
    aggregator_config = config["FL Aggregator"]

    return (
        str(aggregator_config["IP"]),
        int(aggregator_config["Port"]),
        int(client_config["LocalEpochs"]),
        int(client_config["BatchSize"]),
        int(client_config["StepsPerEpoch"]),
        float(client_config["TestSize"])
    )


class StandardClient(fl.client.NumPyClient):
    def __init__(self, client):
        self.X_train, self.X_test = (
            client.X_train,
            client.X_test,
        )

        self.model, self.mse_hist = client.model, client.mse_hist

        self.epochs, self.batch_size, self.steps_per_epoch = (
            client.epochs,
            client.batch_size,
            client.steps_per_epoch,
        )

        self.round_number = 1

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.X_train,
            self.X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            shuffle=True
        )
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        print("ROUND ", self.round_number)
        self.round_number += 1
        self.model.set_weights(parameters)

        predictions = self.model.predict(self.X_test)
        mse_per_sample = mean_squared_error(self.X_test, predictions).numpy()
        mse = mse_per_sample.mean()
        print(mse)
        self.mse_hist.append(mse)
        print(self.mse_hist)
        return 0.0, len(self.X_test), {}


class FLClient:
    def __init__(self, training_data):
        (
            self.aggregator_ip,
            self.aggregator_port,
            self.epochs,
            self.batch_size,
            self.steps_per_epoch,
            self.test_size
        ) = get_config()
        self.X_train, self.X_test, n_features = preprocess_data_client_side(training_data, self.test_size)
        self.model = create_basic_autoencoder(n_features)
        self.mse_hist = []

    def start(self):
        fl.client.start_numpy_client(
            server_address="{aggregator_ip}:{aggregator_port}".format(
                aggregator_ip=self.aggregator_ip, aggregator_port=self.aggregator_port
            ),
            client=StandardClient(self)
        )

    def get_detection_threshold(self):
        predictions = self.model.predict(self.X_train)
        train_mse = mean_squared_error(self.X_train, predictions)

        print(train_mse)
        print(type(train_mse))
        threshold = np.percentile(train_mse, 95)
        print(threshold)

        return threshold

    def get_final_model(self):
        return self.model

    def get_mse_hist(self):
        return self.mse_hist
