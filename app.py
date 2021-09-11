import json
import logging
import os
from flask import Flask, request, abort
from werkzeug.utils import secure_filename

#user defined modules import statements
from src.scripts.dao.database_operations import create_table
import  src.scripts.service.sales_service as sales_service


#declaration flask app for discovering all packages
app = Flask(__name__)

#flask app properties
# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.csv']
app.config['UPLOAD_PATH_TRAIN'] = 'data/uploads/train'
app.config['UPLOAD_PATH_TEST'] = 'data/uploads/test'

#response format
#response_type={"status":true,message:"Success",data:[]}


@app.before_first_request
def create_tables():
    create_table()



@app.route('/user/train/create', methods=['GET', 'POST'])
def add_training_data():
    if request.method == 'GET':
        return {"message":"please hit post request"}
    if request.method == 'POST':
        data=request.get_json(force=True, silent=False, cache=True)
        logging.info(f"post request came with data {data}")
        #insert_a_sale_data(data)
        sales_service.upload_a_train_data_to_db(data)
        return {"status":True,'message':"Successfully store the data into database",data:[]}

@app.route('/user/train/all',methods=["GET"])
def RetrieveList():
    #sales = get_all_data()
    sales = sales_service.get_train_data_from_db()
    sales = [json.loads(x) for x in sales]
    return {"Sales":sales}

###################################### api for train for newly added data  #####################
#request form for getting data
@app.route("/user/train",methods=["POST"])
def train():
    if request.method == 'POST':

        data=request.get_json(force=True, silent=False, cache=True)
        #csv upload or single data upload for training data
        if data['file_flag'] :
            # make sure the file name in file type html form should be file
            uploaded_file = request.files['file']
            filename = secure_filename(uploaded_file.filename)
            if filename != '':
                file_ext = os.path.splitext(filename)[1]
                if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                    abort(400)
                uploaded_file.save(os.path.join(app.config['UPLOAD_PATH_TRAIN'], filename))
                sales_service.load_train_csv_to_db(os.path.join(app.config['UPLOAD_PATH_TRAIN'], filename))
                sales_service.train_model()
                return {"status":True,"message":"Success fully uploaded the data",data:[]}
        else :
            sales_service.upload_a_train_data_to_db(data)
            #a single data can't effect a much so no need to train the model again
            #sales_service.train()
            return {"status":True,'message':"Success fully uploaded the data"}

@app.route("/user/predict",methods=["POST"])
def predict():
    if request.method == 'POST':

        data=request.get_json(force=True, silent=False, cache=True)
        #csv upload or single data upload for training data
        if data['file_flag'] :
            # make sure the file name in file type html form should be file
            uploaded_file = request.files['file']
            filename = secure_filename(uploaded_file.filename)
            if filename != '':
                file_ext = os.path.splitext(filename)[1]
                if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                    abort(400)
                uploaded_file.save(os.path.join(app.config['UPLOAD_PATH_TEST'], filename))
                result=sales_service.predict_sales_csv(os.path.join(app.config['UPLOAD_PATH_TEST'], filename))
                return {"status":True,'message':"Predicted sales are ",data:[result]}
        else :
            result=sales_service.predict_sales(data)

            return {"status":True,'message':"Predicted sales is ",data:[result]}

if __name__=='__main__':
    app.run(debug=True,host='localhost', port=5000)