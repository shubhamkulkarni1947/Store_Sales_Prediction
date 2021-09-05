import json

from flask import Flask, request
import logging

#user defined modules import
from database_operations import insert_a_sale_data,get_all_data,create_table,load_training_data

#declaration flask app for discovering all packages
app = Flask(__name__)


@app.before_first_request
def create_tables():
    create_table()



@app.route('/train/sales/create', methods=['GET', 'POST'])
def add_training_data():
    if request.method == 'GET':
        return {"message":"please hit post request"}
    if request.method == 'POST':
        data=request.get_json(force=True, silent=False, cache=True)
        logging.info(f"post request came with data {data}")

        insert_a_sale_data(data['Item_Identifier'],data['Item_Weight'],data['Item_Fat_Content'],
                            data['Item_Visibility'],data['Item_Type'],data['Item_MRP'],data['Outlet_Identifier'],
                            data['Outlet_Establishment_Year'],data['Outlet_Size'],
                            data['Outlet_Location_Type'],data['Outlet_Type'],data['Item_Outlet_Sales'])

        return "successfully stored the data"


@app.route('/sales/train',methods=["GET"])
def RetrieveList():
    sales = get_all_data()
    sales = [json.loads(x) for x in sales]
    return {"Sales":sales}

@app.route('/sales/predict',methods= ["GET","POST"])
def predict():
    pass

if __name__=='__main__':
    app.run(debug=True,host='localhost', port=5000)