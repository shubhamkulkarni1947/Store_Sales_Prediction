import logging
import json
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import dict_factory

import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET =os.getenv("CLIENT_SECRET")


from models import SalesModelEncoder

cloud_config = {'secure_connect_bundle': 'config/secure-connect-store-sales.zip'}
auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)

def get_session():
    try :
        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        session = cluster.connect()
    except Exception as e :
        raise Exception("Not able to connect to database. Error - {}".format(e))
    return session

def query_executor(query):

    session=get_session()
    try:
        session.row_factory = dict_factory
        logging.info(query)
        result=session.execute(query)
    except Exception as e :
        raise("query execution got failed with error {}".format(e))
    finally:
        session.shutdown()
    return result if result is not None else "query executed successfully"

def create_table():
    query= ("CREATE TABLE IF NOT EXISTS  sales.sales_train (\n"
            "    \"Item_Identifier\" text PRIMARY KEY,\n"
            "    \"Item_Fat_Content\" text,\n"
            "    \"Item_MRP\" decimal,\n"
            "    \"Item_Outlet_Sales\" decimal,\n"
            "    \"Item_Type\" text,\n"
            "    \"Item_Visibility\" decimal,\n"
            "    \"Item_Weight\" decimal,\n"
            "    \"Outlet_Establishment_Year\" int,\n"
            "    \"Outlet_Identifier\" text,\n"
            "    \"Outlet_Location_Type\" text,\n"
            "    \"Outlet_Size\" text,\n"
            "    \"Outlet_Type\" text\n"
            ")")
    query_executor(query)
    logging.info("table created successfully")

def get_train_data():
    sales_train_data = query_executor("select * from sales.sales_train").all()
    sales_train_data=[json.dumps(x,cls=SalesModelEncoder) for x in sales_train_data]
    return sales_train_data

def insert_a_train_data(data):
    query= f"""INSERT INTO sales.sales_train 
    ("Item_Identifier","Item_Weight","Item_Fat_Content","Item_Visibility","Item_Type","Item_MRP",
    "Outlet_Identifier","Outlet_Establishment_Year","Outlet_Size","Outlet_Location_Type",
    "Outlet_Type","Item_Outlet_Sales")
    VALUES 
    ('{data['Item_Identifier']}',{data['Item_Weight']},'{data['Item_Fat_Content']}',{data['Item_Visibility']},
    '{data['Item_Type']}',{data['Item_MRP']},'{data['Outlet_Identifier']}',{data['Outlet_Establishment_Year']},
    '{data['Outlet_Size']}','{data['Outlet_Location_Type']}','{data['Outlet_Type']}',{data['Item_Outlet_Sales']});"""

    query_executor(query)


def load_training_csv_data(filepath):
    filepath="data/raw/Train.csv"
    # TODO need to discuss will programmatically dump the data or use direct feature import cs
    """COPY sales.sales_train FROM 'data/raw/Train.csv' WITH DELIMITER=',' AND HEADER=TRUE"""
    query=f"""COPY sales.sales_train FROM '{filepath}' WITH DELIMITER=',' AND HEADER=TRUE"""
    query_executor(query)

