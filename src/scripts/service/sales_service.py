#import statements
import src.scripts.dao.database_operations as dao
import pandas as pd
import numpy as np
from joblib import load
import os
from pathlib import Path
from .util_script import clean_data, feature_encoding,remove_irrelevant_columns,complete_flow_till_model_creation



# function for training the model
def train_model():
    # TODO : train the model follow complete process
    complete_flow_till_model_creation()


# function for predicting the value
def predict_sales_csv(test_csv_filepath):

    df = pd.read_csv(test_csv_filepath)
    # Transform the dataframe -> cleaning,encoding
    test_df = clean_data(df)
    test_df = feature_encoding(test_df, True)
    test_df = remove_irrelevant_columns(test_df)

    # predicting result after transformation of data
    model_pipe = load('models\model.pkl')
    prediction = model_pipe.predict(test_df)

    # format the prediction by adding it as a column in the current dataframe
    df['Item_Outlet_Sales'] = np.round(prediction, 3)

    # Converting back df to list of dict
    pred_data = df.to_dict('records')
    return pred_data

def predict_sales(data):

    df = pd.DataFrame(data)
    #Transform the dataframe -> cleaning,encoding
    test_df = clean_data(df)
    test_df = feature_encoding(test_df,True)
    test_df = remove_irrelevant_columns(test_df)

    #predicting result after transformation of data
    # model_path = os.path.join(path, '/models/model.pkl')
    model_pipe = load('models\model.pkl')
    prediction = model_pipe.predict(test_df)

    #format the prediction by adding it as a column in the current dataframe
    df['Item_Outlet_Sales'] = np.round(prediction, 3)

    #Converting back df to list of dict
    pred_data = df.to_dict('records')
    return pred_data


# other supporting function


def load_train_csv_to_db(filepath):
    dao.load_training_csv_data(filepath)

# validate the data
def upload_a_train_data_to_db(data):
    print(data)
    for record in data:
        dao.insert_a_train_data(record)


def get_train_data_from_db():
    data = dao.get_train_data()
    return data
