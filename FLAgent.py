import configparser
import json
import requests
import pandas as pd
from datetime import datetime
from copy import copy
from kafka import KafkaConsumer
from FLClient import FLClient
from preprocessor import preprocess_data_client_side

CONFIG_FILE = "resources/fl_agent.ini"

def get_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    agent_config = config["FL Agent"]
    aggregator_config = config["FL Aggregator"]

    return (
        int(agent_config["TrainingThreshold"]),
        str(agent_config["KafkaBrokerIP"]),
        int(agent_config["KafkaBrokerPort"]),
        str(agent_config["KafkaConsumerTopic"]),
        str(aggregator_config["IP"]),
        str(aggregator_config["APIPort"]),
    )

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
    training_threshold, broker_ip, broker_port, consumer_topic, aggregator_ip, aggregator_api_port = \
        get_config()

    consumer = create_kafka_consumer(consumer_topic, broker_ip, broker_port)

    normal_flows = []
    current_model = None 

    for flow_info in consumer:
        store_flow(flow_info, normal_flows)
            
        if len(normal_flows) > training_threshold:
            current_model = federated_training(normal_flows, aggregator_ip, aggregator_api_port)
            normal_flows.clear()
            current_model.save('model-{ts}'.format(ts=datetime.now()))

if __name__ == '__main__':
    main()
