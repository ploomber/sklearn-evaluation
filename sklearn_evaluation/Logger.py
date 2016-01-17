from pymongo import MongoClient
from utils import get_model_name
import datetime

class Logger:
    def __init__(self, host, db, collection):
        #Start connection
        client = MongoClient(host)
        self.collection = client[db][collection]
    def log_model(self, model, **keywords):
        params = model.get_params()
        name = get_model_name(model)
        dt = datetime.datetime.utcnow() 
        model = {'name': name, 'parameters': params, 'datetime': dt}
        model.update(keywords)
        inserted_id = self.collection.insert_one(model).inserted_id
        return inserted_id.str