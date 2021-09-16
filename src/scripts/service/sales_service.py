# import statements
import src.scripts.dao.database_operations as dao
import pandas as pd
import numpy as np
from joblib import load
from flask import json
import models as models
import os
from pathlib import Path
from .util_script import clean_data, feature_encoding, remove_irrelevant_columns, complete_flow_till_model_creation


# function for training the model
def train_model():
    # TODO : train the model follow complete process
    complete_flow_till_model_creation()


def predict_sales(data: list, filename: str, isCsvFile: bool):
    orig_df: pd.DataFrame
    df: pd.DataFrame

    if isCsvFile:
        df = pd.read_csv('data/uploads/pred/' + filename)
    else:
        df = pd.DataFrame(data)

    orig_df = df

    # Transform the dataframe -> cleaning,encoding
    test_df = clean_data(df)
    test_df = feature_encoding(test_df, True)
    test_df = remove_irrelevant_columns(test_df)

    # predicting result after transformation of data
    # model_path = os.path.join(path, '/models/model.pkl')
    model_pipe = load('models/model.pkl')
    prediction = model_pipe.predict(test_df)

    # format the prediction by adding it as a column in the current dataframe
    orig_df['Item_Outlet_Sales'] = np.round(prediction, 3)

    # Converting back df to list of dict
    pred_data = orig_df.to_dict('records')
    return pred_data


# function for convering training log to dict

def get_train_log() -> list:
    lst = []
    with open('src/other/logs/train_log.txt') as f:
        st = f.read()
        st = st.split('*')
        for i in range(len(st)):
            if i % 2 != 0:
                stt = st[i].split('\n')
                dct = {}
                for j in range(1, len(stt) - 1):
                    split_text = stt[j].split(': ')
                    key = split_text[0].strip()
                    val = split_text[1].strip()
                    dct[key] = val
                lst.append(dct)
    lst = [json.dumps(x, cls=models.SalesModelEncoder) for x in lst]
    return lst


# function for checking for duplicates and adding/incrementing ID

def check_duplicate_and_increment_id(data_dict: dict, data_csv_file_name: str, isCSVFile: bool) -> list:
    train_df = pd.read_csv('data/raw/Train.csv')
    new_df: pd.DataFrame
    if isCSVFile:
        new_df = pd.read_csv('data/uploads/train/' + data_csv_file_name)
    else:
        new_df = pd.DataFrame(data_dict)

    # getting last_id from train_df
    last_id = max(train_df['id'])
    print(last_id)

    train_df = train_df.drop(columns=['id'])
    concat_df: pd.DataFrame = pd.concat([train_df, new_df]).reset_index(drop=True)

    len_duplicate: int = len(concat_df[concat_df.duplicated()])
    is_duplicate_present: bool = len_duplicate > 0
    # Non duplicates data from new uploaded data (this will return non duplicate data's) if all
    # duplicate it will return 0 rows
    non_duplicates_data: pd.DataFrame = pd.merge(new_df, train_df, indicator=True, how='outer'). \
        query('_merge=="left_only"'). \
        drop('_merge', axis=1).reset_index(drop=True)

    print(new_df)
    print(non_duplicates_data)

    # Case: If all duplicates present then no need to do anything
    if len(non_duplicates_data) == 0:
        # No need to train as all data are already present in Train.csv file
        return []

    # Case: If 1 or more non duplicate row present
    else:
        # Concatenating non_duplicate_data to train_df and indexing
        final_df: pd.DataFrame = pd.concat([train_df, non_duplicates_data]).reset_index(drop=True)
        final_df = final_df.reset_index()
        final_df = final_df.rename(columns={'index': 'id'})

        # saving this final_df inside data/raw/Train.csv
        final_df.to_csv('data/raw/Train.csv', index=False)

        # Adding indexing (id) to our non duplicate data
        non_duplicates_data['id'] = np.arange(last_id + 1, last_id + len(non_duplicates_data) + 1)
        print('prev id: {}, new_ids: {}'.format(last_id, non_duplicates_data['id']))
        print(non_duplicates_data.columns)
        # convert non_duplicates_data to dict before passing it to db
        non_duplicates_data_dict = non_duplicates_data.to_dict('records')

        return non_duplicates_data_dict


# other supporting function

def load_train_csv_to_db(filepath):
    dao.load_training_csv_data(filepath)


# validate the data
def upload_a_train_data_to_db(data):
    # print(data)
    for record in data:
        dao.insert_a_train_data(record)


def get_train_data_from_db():
    data = dao.get_train_data()
    return data


def get_data_by_id(ID: int):
    data = dao.get_data_by_id(ID)
    return data
