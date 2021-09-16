from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import logging
from cassandra.query import dict_factory
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
cloud_config = {'secure_connect_bundle': 'config/secure-connect-store-sales.zip'}
auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)

def get_session():
    try:
        session = cluster.connect()
    except Exception as e:
        raise Exception("query execution got failed with error {}".format(e))
    return session

def query_executor(query):
    session = get_session()
    print('Connected to DB!')
    try:
        session.row_factory = dict_factory
        logging.info(query)
        result = session.execute(query)
        # print(result.all())
        return result
    except Exception as e:
        raise Exception("query execution got failed with error {}".format(e))
    finally:
        return result if result is not None else "query executed successfully"
        session.shutdown()

data = query_executor('select * from sales.sales_train;').all()
print(len(data))