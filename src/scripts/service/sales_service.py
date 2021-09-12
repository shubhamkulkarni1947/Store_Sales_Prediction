#import statements
import src.scripts.dao.database_operations as dao
import pandas as pd
from joblib import load
import os
from pathlib import Path
from .util_script import clean_data, feature_encoding ,complete_flow_till_model_creation



# function for training the model
def train_model():
    #TODO : train the model follow complete process
    complete_flow_till_model_creation()

# function for predicting the value
def predict_sales_csv(test_csv_filepath):
    # TODO :load test file as df
    #predict the csv file data and return it as json

    df = pd.read_csv(test_csv_filepath)
    # Transform the dataframe -> cleaning,encoding
    df = clean_data(df)
    df = feature_encoding(df)

    # predicting result after transformation of data
    model_pipe = load('../../../models/model.pkl')
    prediction = model_pipe.predict(df)

    # format the prediction by adding it as a column in the current dataframe
    df['Item_Outlet_Sales'] = prediction

    # Converting back df to list of dict
    pred_data = df.to_dict('records')
    return pred_data
    pass

def predict_sales(data):

    df = pd.DataFrame(data)
    #Transform the dataframe -> cleaning,encoding
    df = clean_data(df)
    df = feature_encoding(df)

    #predicting result after transformation of data
    path = Path.cwd()
    model_path = os.path.join(path, 'models\model.pkl')
    model_pipe = load(model_path)
    prediction = model_pipe.predict(df)

    #format the prediction by adding it as a column in the current dataframe
    df['Item_Outlet_Sales'] = prediction

    #Converting back df to list of dict
    pred_data = df.to_dict('records')
    print(pred_data)
    return pred_data


# other supporting function


def load_train_csv_to_db(filepath):
    dao.load_training_csv_data(filepath)

# validate the data
def upload_a_train_data_to_db(data):
    dao.insert_a_train_data(data)

def get_train_data_from_db():
    data=dao.get_train_data()
    return data