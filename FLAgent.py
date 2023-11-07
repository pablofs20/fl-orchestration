import configparser
import json
import requests
import pandas as pd
import threading
from datetime import datetime
from copy import copy
from kafka import KafkaConsumer
from flask import Flask, jsonify, send_file
from FLClient import FLClient
from preprocessor import preprocess_data_client_side

CONFIG_FILE = "resources/fl_agent.ini"


def get_config(section, parameter):
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    return config[section][parameter]

def create_kafka_consumer(topic, broker_ip, broker_port):
    consumer = KafkaConsumer(topic, group_id=None,
                        auto_offset_reset='earliest',
                        bootstrap_servers=['{ip}:{port}'.format(ip=broker_ip, port=broker_port)])
    
    return consumer

def store_flow(flow_info, flow_list):
    flow_info = flow_info.value.decode('utf-8')
    flow_list.append(json.loads(flow_info))

def federated_training(flow_list, aggregator_ip, aggregator_api_port):
    # Create pandas dataframe from flow list
    column_names = flow_list[0].keys()
    training_data = pd.DataFrame(flow_list, columns=column_names)

    # Remove columns according to aggregator's variable set
    aggregator_api_uri = "http://{aggregator_ip}:{aggregator_api_port}/getVariables".format(aggregator_ip=aggregator_ip, aggregator_api_port=aggregator_api_port)
    response = requests.get(aggregator_api_uri)
    aggregator_selected_columns = list(response.json())

    og_columns = list(training_data.columns)
    difference = list(set(og_columns) - set(aggregator_selected_columns))
    training_data = training_data.drop(columns=difference, axis=1)

    # Create and start the FL client
    fl_client = FLClient(training_data)
    fl_client.start()

    return fl_client.get_final_model()

def main():
    global current_model

    training_threshold = int(get_config("FL Agent", "TrainingThreshold"))
    broker_ip = get_config("FL Agent", "KafkaBrokerIP")
    broker_port = int(get_config("FL Agent", "KafkaBrokerPort"))
    consumer_topic = get_config("FL Agent", "KafkaConsumerTopic")
    aggregator_ip = get_config("FL Aggregator", "IP")
    aggregator_api_port = int(get_config("FL Aggregator", "APIPort"))

    consumer = create_kafka_consumer(consumer_topic, broker_ip, broker_port)

    normal_flows = []

    for flow_info in consumer:
        store_flow(flow_info, normal_flows)
            
        if len(normal_flows) > training_threshold:
            final_model = federated_training(normal_flows, aggregator_ip, aggregator_api_port)

            with lock:
                current_model = final_model

            normal_flows.clear()
            current_model.save('model-{ts}'.format(ts=datetime.now()))

if __name__ == '__main__':
    current_model = None
    
    threading.Thread(target=main).start()

    app = Flask(__name__)

    lock = threading.Lock()
    
    @app.route("/getModel", methods=["GET"])
    def get_model():
        global current_model

        with lock:
            if current_model is not None:
                serialized_model = current_model.to_json()
                return jsonify({"model": serialized_model})
            else:
                return jsonify({"error": "Model not available yet"})

    api_port = int(get_config("FL Agent", "APIPort"))

    app.run(host='localhost', port=api_port)
