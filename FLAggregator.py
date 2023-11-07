import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import flwr as fl
import configparser
import pandas as pd
import threading
from flask import Flask, jsonify
from preprocessor import preprocess_data_server_side
from model_creator import create_basic_autoencoder

CONFIG_FILE = 'resources/fl_aggregator.ini'

def get_config(parameter):
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    return config['FL Aggregator'][parameter]

def main():
    global variable_set
    # Load general dataset
    BS1_benign = pd.read_csv("../datasets/BTS_1_benign.csv")
    BS2_benign = pd.read_csv("../datasets/BTS_2_benign.csv")

    general_dataset = pd.concat([BS1_benign, BS2_benign])

    # Preprocess it
    general_dataset = preprocess_data_server_side(general_dataset)

    # # Get variable set after preprocessing
    with lock:
        variable_set = list(general_dataset.columns)
        print(variable_set)

    # # Create model
    model = create_basic_autoencoder(len(general_dataset.columns))

    # # Get model weights as a list of NumPy ndarray's
    weights = model.get_weights()

    # # Serialize ndarrays to `Parameters`
    parameters = fl.common.ndarrays_to_parameters(weights)
    
    rounds = int(get_config("Rounds"))
    min_clients = int(get_config("MinClients"))
    ip = get_config("IP")
    port = get_config("Port")

    strategy = fl.server.strategy.FedAvg(
            fraction_fit = 1,
            min_fit_clients = min_clients,
            min_available_clients = min_clients,
            initial_parameters=parameters
    )

    while True:
        fl.server.start_server(server_address = '{ip}:{port}' \
            .format(ip = ip, port = port),
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy
        )

if __name__ == '__main__':
    variable_set = None
    
    threading.Thread(target=main).start()

    app = Flask(__name__)

    lock = threading.Lock()
    
    #@app.route("/start", methods=["GET"])
    #def start():
     #   threading.Thread(target=main).start()

      #  response = {"Message": "FL Aggregator has been started successfully"}
      #  return response, 201

    @app.route("/getVariables", methods=["GET"])
    def get_variables():
        global variable_set

        with lock:
            return jsonify(variable_set), 201

    api_port = get_config("APIPort")

    app.run(host='localhost', port=api_port)

