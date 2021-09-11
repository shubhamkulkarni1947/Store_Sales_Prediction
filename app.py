<<<<<<< Updated upstream
# using flask_restful
from flask import Flask, jsonify, request
from flask_restful import Resource, Api

# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)
=======
import json
from flask import Flask, request
from werkzeug.utils import secure_filename
import logging
import os

# user defined modules import
from database_operations import insert_a_sale_data, get_all_data, create_table, load_training_data

# declaration flask app for discovering all packages
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.relpath(__file__))
file_folder = '../static/files/'
UPLOAD_FOLDER = file_folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
>>>>>>> Stashed changes


class Hello(Resource):

    # corresponds to the GET request.
    # this function is called whenever there
    # is a GET request for this resource
    def get(self):
        return jsonify({'message': 'hello world'})

<<<<<<< Updated upstream
    # Corresponds to POST request
    def post(self):
        data = request.get_json()  # status code
        return jsonify({'data': data}), 201


# another resource to calculate the square of a number
class Square(Resource):

    def get(self, num):
        return jsonify({'square': num ** 2})


# adding the defined resources along with their corresponding urls
api.add_resource(Hello, '/')
api.add_resource(Square, '/square/<int:num>')

# driver function
if __name__ == '__main__':
    app.run(debug=True)
=======
@app.route('/sales/data', methods=["GET"])
def retrieveAllData():
    sales = get_all_data()
    sales = [json.loads(x) for x in sales]
    return {"status": True, "message": "success", "sales": sales}

@app.route('/sales/data/<data_id>', methods=["GET"])
def getDataById(data_id):
    #Get data By Id from dao service
    return {"status": True, "message": data_id}

# Routes for User Prediction

@app.route('/sales/prediction', methods=["GET", "POST"])
def getPredictedSales():
    if request.method == "POST":
        if request.files:
            file = request.files['file']
            # file.save('static/files/'+secure_filename(file.filename))
            return {"status" : True, "status": "File"}
        else:
            print(request.form)
            return {"status" : True,"status": "Form Data"}
    else:
        return {"status":False, "message":"Method Not Supported"}


# Routes for adding new data and retraining the model

@app.route('/sales/add', methods=['POST'])
def addTrainingData():
    if request.method == "POST":
        data = request.get_json(force=True, silent=False, cache=True)
        logging.info(f"post request came with data {data}")

        insert_a_sale_data(data['Item_Identifier'], data['Item_Weight'], data['Item_Fat_Content'],
                           data['Item_Visibility'], data['Item_Type'], data['Item_MRP'], data['Outlet_Identifier'],
                           data['Outlet_Establishment_Year'], data['Outlet_Size'],
                           data['Outlet_Location_Type'], data['Outlet_Type'], data['Item_Outlet_Sales'])

        return {"status":True, "message":"Successfully stored Data."}
    else:
        return {"status": False, "message": "Method Not Supported"}


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
>>>>>>> Stashed changes
