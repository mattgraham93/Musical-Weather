import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd

import cloud_driver

# Replace these with your Google Cloud project ID and the names of your secrets
project_id = "musical-weather"
secret_id_cert = "mongo_cert"
secret_id_auth = "mongo_musical_weather"

# Access the secrets
cert_value = cloud_driver.access_secret_version(project_id, secret_id_cert)
auth_value = cloud_driver.access_secret_version(project_id, secret_id_auth)

def get_client():
    # Save the certificate content to a file
    with open('certificate.pem', 'w') as f:
        f.write(cert_value)

    # Pass the path to the file as the tlsCertificateKeyFile argument
    return MongoClient(auth_value, tls=True, tlsCertificateKeyFile='certificate.pem', server_api=ServerApi('1'))        

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
    data = list(collection.find())
    
    # Remove '_id' key from each dictionary
    for item in data:
        item.pop('_id', None)
    
    return data