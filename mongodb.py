import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd

filename = r"D:\Backup\repos\auth.json"
cert_file = r"D:\Backup\repos\musicalweather.pem"

with open(filename) as file:
    data = json.load(file)
    mongo_conn_st = data['musical_weather']

def get_client():
    return MongoClient(mongo_conn_st, tls=True, tlsCertificateKeyFile=cert_file, server_api=ServerApi('1'))

def store_collection(database_name, collection_name, data):
    mongo_client = get_client()
    db = mongo_client.client[database_name]
    collection = db[collection_name]
    
    # Check if data is a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.to_dict('records')  # Convert DataFrame to list of dictionaries
    collection.insert_many(data)  # data is now guaranteed to be a list of dictionaries
    
    # https://stackoverflow.com/questions/20167194/insert-a-pandas-dataframe-into-mongodb-using-pymongo
    return db

def get_stored_data(database_name, collection_name):
    mongo_client = get_client()
    db = mongo_client.client[database_name]
    collection = db[collection_name]
    return list(collection.find())
