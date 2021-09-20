import logging
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import dump, load
import src.scripts.dao.database_operations as dao

log = logging.getLogger('util_script')

pd.options.mode.chained_assignment = None

lEncIT_mapping = {'Baking Goods': 0,
                  'Breakfast': 1,
                  'Canned': 2,
                  'Dairy': 3,
                  'Drinks': 4,
                  'Frozen Foods': 5,
                  'Fruits and Vegetables': 6,
                  'Health and Hygiene': 7,
                  'Household': 8,
                  'Meat': 9,
                  'Others': 10,
                  'Snack Foods': 11,
                  'Starchy Foods': 12}

lEncOT_mapping = {'Grocery Store': 0,
                  'Supermarket Type1': 1,
                  'Supermarket Type2': 2,
                  'Supermarket Type3': 3}


# ########################## complete flow #########################################

def complete_flow_till_model_creation():
    train_df = get_train_df()
    train_df = clean_data(train_df, isTrain=True)
    train_df = feature_encoding(train_df, isPrediction=False)
    train_df = remove_irrelevant_columns(train_df)
    train_df = predict_missing_values_Outlet_size(train_df, isTrain=True)
    score_data = train_model(train_df)
    return score_data


# ######################### load data source ##############################
def get_train_df():
    log.info("Getting ALl training data from DB")
    #    return pd.read_csv('../../../data/raw/Train.csv')
    # train_data = dao.get_train_data()
    # sales = [json.loads(x) for x in train_data]
    sales = pd.read_csv('data/raw/Train.csv')
    return sales


def get_test_df():
    return pd.read_csv('data/raw/Test.csv')


# ######################### clean data source ##############################
def clean_data(df: pd.DataFrame, isTrain: bool):
    log.info("Cleaning/Imputing Data")
    # ######################### handle_missing_value ############################

    if isTrain:
        with open('models/imputation_values.txt', 'w') as f:
            f.write('mean_item_wight={}\n'.format(df['Item_Weight'].mean()))
            f.write('median_item_visibility={}\n'.format(df['Item_Visibility'].median()))

        # imputing Missing Value for Item_Weight with Mean
        df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)

        df['Item_Visibility'].replace({0: df['Item_Visibility'].median()}, inplace=True)

        # ############################# Removing duplicates##########################

        df['Item_Fat_Content'].replace({'LF': 'Low Fat',
                                        'low fat': 'Low Fat',
                                        'reg': 'Regular'}, inplace=True)

        return df
    else:
        df['Item_Weight'].replace({0: np.nan}, inplace=True)
        df['Outlet_Size'].replace({"": np.nan}, inplace=True)
        with open('models/imputation_values.txt', 'r') as f:
            imp_values = f.read().strip().split("\n")
            mean_item_weight = float(imp_values[0].split("=")[1])
            median_item_visibility = float(imp_values[1].split("=")[1])

            df['Item_Weight'].fillna(mean_item_weight, inplace=True)
            df['Item_Visibility'].replace({0: median_item_visibility}, inplace=True)
            df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}, inplace=True)
        return df


def feature_encoding(df: pd.DataFrame, isPrediction: bool):
    log.info("Feature Encoding the data")
    global lEncIT_mapping
    global lEncOT_mapping

    df['Item_Type'].replace(
        {'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Breads': 'Baking Goods', 'Seafood': 'Meat'}, inplace=True)

    # Binary Encoding
    df['Item_Fat_Content'].replace({'Low Fat': 0, 'Regular': 1}, inplace=True)

    nominal_features = ['Outlet_Type', 'Item_Type']
    prefixes = ['out_type', 'item_type']

    if isPrediction:
        df['Item_Type'].replace(lEncIT_mapping, inplace=True)
        df['Outlet_Type'].replace(lEncOT_mapping, inplace=True)
    else:
        lEncIT = LabelEncoder()
        lEncOT = LabelEncoder()

        df.loc[:, 'Outlet_Type'] = lEncOT.fit_transform(df['Outlet_Type'])
        df.loc[:, 'Item_Type'] = lEncIT.fit_transform(df['Item_Type'])
        lEncIT_mapping = dict(zip(lEncIT.classes_, lEncIT.transform(lEncIT.classes_)))
        lEncOT_mapping = dict(zip(lEncOT.classes_, lEncOT.transform(lEncOT.classes_)))

    # Encoding Ordinal columns
    outlet_size_ord = {'Small': 0, 'Medium': 1, 'High': 2}
    out_loc_ord = {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}

    df = ord_enc(df, 'Outlet_Size', outlet_size_ord)

    df = ord_enc(df, 'Outlet_Location_Type', out_loc_ord)

    # # Deriving new column called Years_Since_Established from Establishment Year
    df['Years_Since_Established'] = df['Outlet_Establishment_Year'].apply(lambda x: 2021 - x)

    return df


# Label Encoding

def label_enc(df: pd.DataFrame, cols):
    pass


# One_hot encoding nominal variables

def one_hot(df, columns, prefixes):
    df = df.copy()
    for column, prefix in zip(columns, prefixes):
        dummies = pd.get_dummies(df[column], prefix=prefix, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


# Ordinal Encoding

def ord_enc(df, col, ord_var):
    df = df.copy()
    df[col].replace(ord_var, inplace=True)
    return df


def remove_irrelevant_columns(df: pd.DataFrame):
    log.info("Removing irrelevant columns")
    df.drop(columns=['Item_Identifier', 'Outlet_Identifier'], axis=1, inplace=True)  # ,'Outlet_Identifier'
    if 'id' in df.columns:
        df.drop(columns=['id'], axis=1, inplace=True)
    return df


def predict_missing_values_Outlet_size(df: pd.DataFrame, isTrain: bool):
    log.info('Predicting missing values for Outlet_Size')
    if isTrain:

        df['Outlet_Size'].replace({'None': np.nan}, inplace=True)
        out_train_pred_df = df[df['Outlet_Size'].isna()]
        # out_test_pred_df = test_df[test_df['Outlet_Size'].isna()]
        out_train_df = df[~df['Outlet_Size'].isna()]  # for training
        out_train_df['Outlet_Size'].replace({'Small': 0, 'Medium': 1, 'High': 2}, inplace=True)
        # out_train_df.drop(columns=['Item_Identifier','Outlet_Identifier'],inplace=True)
        X = out_train_df.drop(columns=['Outlet_Size', 'Item_Outlet_Sales'])
        y = out_train_df['Outlet_Size']
        # print(y.value_counts())
        trainX, testX, trainY, testY = train_test_split(X, y, random_state=22, test_size=0.2)
        rf_model = RandomForestClassifier(random_state=2)
        rf_model.fit(trainX, trainY)

        # pred = cat_model.predict(testX)

        out_train_pred = rf_model.predict(out_train_pred_df.drop(columns=['Outlet_Size', 'Item_Outlet_Sales']))
        # print(out_train_pred)
        out_train_pred_df.loc[:, 'Outlet_Size'] = out_train_pred
        # out_test_pred = rf_model.predict(out_test_pred_df.drop(columns=['Outlet_Size']))
        # out_test_pred_df['Outlet_Size'] = out_test_pred

        df.dropna(inplace=True)
        # test_df.dropna(inplace=True)

        df = pd.concat([df, out_train_pred_df])
        # test_df = pd.concat([test_df, out_test_pred_df])
        dump(rf_model, 'models/outlet_size_model.pkl')
        return df
    else:
        # print(type(df['Outlet_Size'][0]))
        rf_model = load('models/outlet_size_model.pkl')
        pred_out_size_df = df[df['Outlet_Size'].isna()]
        df = df[~df['Outlet_Size'].isna()]
        if len(pred_out_size_df > 0):
            pred_values = rf_model.predict(pred_out_size_df.drop(columns=['Outlet_Size']))
            pred_out_size_df.loc[:, 'Outlet_Size'] = pred_values
            df = pd.concat([df, pred_out_size_df])
            return df
        else:
            return df


# Data Format (dictionary): {'ID':'Axsd34','Outlet_Size':'Medium',...}
# def createDataFrameUsingForm(data: dict) -> pd.DataFrame:
#     df = pd.DataFrame.from_dict(data,orient='index').T
#     return df

def train_model(train_df):
    log.info("Training the model.")
    X = train_df.drop(columns=['Item_Outlet_Sales'], axis=1)
    y = train_df['Item_Outlet_Sales']

    trainX, testX, trainY, testY = train_test_split(X, y, random_state=42, test_size=0.25)

    clf = Pipeline(
        [('cat_reg', CatBoostRegressor(random_state=2, iterations=3000,
                                       learning_rate=0.002,
                                       depth=6, silent=True,
                                       allow_writing_files=False))])
    clf.fit(trainX, trainY)

    # print('Trining R2 Score: {}'.format(clf.score(trainX, trainY)))

    pred = clf.predict(testX)
    model_name = 'CatBoostRegressor'

    score_data = predictionResult(testY, pred, model_name)
    # logging training scores to file
    logToFile('src/other/logs/train_log.txt', score_data)
    dump(clf, 'models/model.pkl')
    return score_data


def predictionResult(testY, pred, model_name):
    score = np.round((r2_score(testY, pred) * 100), 2)
    mae = np.round(mean_absolute_error(testY, pred), 2)
    mse = np.round(mean_squared_error(testY, pred), 2)
    rmse = np.round(np.sqrt(mse), 2)
    model_acc_scores = {'r2_accuracy_score': score, 'mae_score': mae, 'mse_score': mse, 'rmse_score': rmse,
                        'model_name': model_name}
    return model_acc_scores


# Log train data to file

def logToFile(path: str, data: dict):
    # For logging training scores to file
    log.info("Logging training record to file.")
    with open(path, 'a') as logfile:
        currTimestamp = datetime.datetime.now()
        logfile.write('---------------------------------------------------*\n')
        logfile.write('TimeStamp: {}\n'.format(currTimestamp))
        logfile.write('Model Name: {}\n'.format(data['model_name']))
        logfile.write('R2 Score: {}%\n'.format(data['r2_accuracy_score']))
        logfile.write('Mean Abs Error: {}\n'.format(data['mae_score']))
        logfile.write('Mean Sq Error: {}\n'.format(data['mse_score']))
        logfile.write('Root Mean Sq Error: {}\n'.format(data['rmse_score']))
        logfile.write('*-----------------------------------------------------\n')
