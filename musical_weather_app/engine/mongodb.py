import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import os

try:
    # Try relative import first
    from . import cloud_driver
except ImportError:
    # Fallback for when running as script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine import cloud_driver

# Replace these with your Google Cloud project ID and the names of your secrets
project_id = "musical-weather"
secret_id_cert = "mongo_cert"
secret_id_auth = "mongo_musical_weather"

# Access the secrets with error handling
cert_value = None
auth_value = None
MONGODB_AVAILABLE = False

try:
    cert_value = cloud_driver.access_secret_version(project_id, secret_id_cert)
    auth_value = cloud_driver.access_secret_version(project_id, secret_id_auth)
    MONGODB_AVAILABLE = True
    print("MongoDB credentials loaded successfully")
except Exception as e:
    print(f"Warning: Could not load MongoDB credentials: {e}")
    print("MongoDB functionality will be disabled")

def get_client():
    if not MONGODB_AVAILABLE:
        raise Exception("MongoDB credentials not available")
        
    import tempfile
    import os
    
    # Use a temporary file for the certificate in production environments
    if os.environ.get('GAE_ENV') or os.environ.get('K_SERVICE'):
        # In production, use a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write(cert_value)
            cert_file_path = f.name
    else:
        # In development, use a regular file
        cert_file_path = 'certificate.pem'
        with open(cert_file_path, 'w') as f:
            f.write(cert_value)

    # Pass the path to the file as the tlsCertificateKeyFile argument
    return MongoClient(auth_value, tls=True, tlsCertificateKeyFile=cert_file_path, server_api=ServerApi('1'))        

def store_collection(database_name, collection_name, data):
    if not MONGODB_AVAILABLE:
        print(f"Warning: Cannot store collection {collection_name} - MongoDB not available")
        return None
        
    try:
        mongo_client = get_client()
        db = mongo_client.client[database_name]
        collection = db[collection_name]
        
        # Check if data is a DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')  # Convert DataFrame to list of dictionaries
        collection.insert_many(data)  # data is now guaranteed to be a list of dictionaries
        
        # https://stackoverflow.com/questions/20167194/insert-a-pandas-dataframe-into-mongodb-using-pymongo
        return db
    except Exception as e:
        print(f"Error storing collection {collection_name}: {e}")
        return None

def get_stored_data(database_name, collection_name):
    if not MONGODB_AVAILABLE:
        print(f"Warning: Cannot retrieve collection {collection_name} - MongoDB not available")
        return []
        
    try:
        mongo_client = get_client()
        db = mongo_client.client[database_name]
        collection = db[collection_name]
        data = list(collection.find())
        
        # Remove '_id' key from each dictionary
        for item in data:
            item.pop('_id', None)
        
        return data
    except Exception as e:
        print(f"Error retrieving collection {collection_name}: {e}")
        return []