import logging
import json
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import dict_factory
import cassandra

import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv("CLIENT_SECRET")


from models import SalesModelEncoder

cloud_config = {'secure_connect_bundle': 'config/secure-connect-store-sales.zip'}
auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)

def get_session():
    # print('cassandra driver version: %s ' % str(cassandra.__version_info__))
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider,protocol_version=4)
    session = cluster.connect()
    return session

def insert_a_sale_data(Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year,Outlet_Size,Outlet_Location_Type,Outlet_Type,Item_Outlet_Sales):
    # session=get_session()
    query= f"""INSERT INTO sales.sales_train ("Item_Identifier","Item_Weight","Item_Fat_Content","Item_Visibility","Item_Type","Item_MRP",
    "Outlet_Identifier","Outlet_Establishment_Year","Outlet_Size","Outlet_Location_Type","Outlet_Type","Item_Outlet_Sales")
    VALUES ('{Item_Identifier}',{Item_Weight},'{Item_Fat_Content}',{Item_Visibility},'{Item_Type}',{Item_MRP},'{Outlet_Identifier}',{Outlet_Establishment_Year},
    '{Outlet_Size}','{Outlet_Location_Type}','{Outlet_Type}',{Item_Outlet_Sales});"""
    # print(query)
    logging.info(query)
    query_executor(query)

def get_all_data():
    sales_train_data = query_executor("select * from sales.sales_train").all()
    # print(sales_train_data)
    sales_train_data=[json.dumps(x,cls=SalesModelEncoder) for x in sales_train_data]
#     print(sales_train_data)
    return sales_train_data

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

def load_training_data():
    # TODO need to discuss will programmatically dump the data or use direct feature import cs
    """COPY sales.sales_train FROM 'data/raw/Train.csv' WITH DELIMITER=',' AND HEADER=TRUE"""
    pass

def query_executor(query):
    session=get_session()
    session.row_factory = dict_factory
    result=session.execute(query)
    session.shutdown()
    return result if result is not None else "query executed successfully"
