import logging
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore, boxcox, probplot
import seaborn as sns
from statsmodels.api import OLS
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, classification_report, confusion_matrix
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,classification_report,confusion_matrix
import warnings
from joblib import dump,load

# ########################## complete flow #########################################

def complete_flow_till_model_creation():
    train_df=get_train_df()
    test_df=get_test_df()
    resulted_dict=clean_data(train_df,test_df)
    resulted_dict=feature_encoding(resulted_dict['train_df'],resulted_dict['test_df'])
    resulted_dict=remove_irrelevant_columns(resulted_dict['train_df'],resulted_dict['test_df'])
    resulted_dict=predict_missing_values_Outlet_size(resulted_dict['train_df'],resulted_dict['test_df'])
    training_model=train_model(resulted_dict['train_df'])

# ######################### load data source ##############################
def get_train_df():
    return pd.read_csv('../../../data/raw/Train.csv')

def get_test_df():
    return pd.read_csv('../../../data/raw/Test.csv')


# ######################### clean data scource ##############################
def clean_data(df: pd.DataFrame):

# ######################### handle_missing_value ############################

    # imputing Missing Value for Item_Weight with Mean
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)

    #TODO applying tree based model to predict the value because null are larger in numbers
    # print(train_df['Outlet_Size'].value_counts())

    df['Item_Visibility'].replace({0: df['Item_Visibility'].median()}, inplace=True)


# ############################# Removing duplicates##############################################

    df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}, inplace=True)

    return df

def feature_encoding(df: pd.DataFrame):

    df['Item_Type'].replace(
        {'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Breads': 'Baking Goods', 'Seafood': 'Meat'}, inplace=True)

    # Binary Encoding
    df['Item_Fat_Content'].replace({'Low Fat': 0, 'Regular': 1}, inplace=True)

    nominal_features = ['Outlet_Type', 'Item_Type', 'Outlet_Identifier']
    prefixes = ['out_type', 'item_type', 'out_id']

    df = one_hot(df, nominal_features, prefixes)

    # Encoding Ordinal columns
    outlet_size_ord = {'Small': 0, 'Medium': 1, 'High': 2}
    out_loc_ord = {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}

    df = ord_enc(df, 'Outlet_Size', outlet_size_ord)

    df = ord_enc(df, 'Outlet_Location_Type', out_loc_ord)

    # # Deriving new column called Years_Since_Established from Establishment Year
    df['Years_Since_Established'] = df['Outlet_Establishment_Year'].apply(lambda x: 2021 - x)


    return df

#One_hot encoding nominal variables

def one_hot(df,columns,prefixes):
    df = df.copy()
    for column,prefix in zip(columns,prefixes):
        dummies = pd.get_dummies(df[column],prefix=prefix,drop_first=True)
        df = pd.concat([df,dummies],axis=1)
        df = df.drop(column,axis=1)
    return df

#Ordinal Encoding

def ord_enc(df,col,ord_var):
    df = df.copy()
    df[col].replace(ord_var,inplace=True)
    return df

def remove_irrelevant_columns(df: pd.DataFrame):
    df.drop(columns=['Item_Identifier'], axis=1, inplace=True)  # ,'Outlet_Identifier'
    return df


def predict_missing_values_Outlet_size(df: pd.DataFrame):
    out_train_pred_df = df[df['Outlet_Size'].isna()]
    # out_test_pred_df = test_df[test_df['Outlet_Size'].isna()]
    out_train_df = df[~df['Outlet_Size'].isna()]  # for training
    out_train_df.isna().sum()
    # out_train_df['Outlet_Size'] = out_train_df['Outlet_Size'].replace({'Small':0,'Medium':1,'High':2})
    # out_train_df.drop(columns=['Item_Identifier','Outlet_Identifier'],inplace=True)
    X = out_train_df.drop(columns=['Outlet_Size', 'Item_Outlet_Sales'])
    y = out_train_df['Outlet_Size']
    trainX, testX, trainY, testY = train_test_split(X, y, random_state=22, test_size=0.2)
    rf_model = RandomForestClassifier(random_state=2)
    rf_model.fit(trainX, trainY)

    # pred = cat_model.predict(testX)

    out_train_pred = rf_model.predict(out_train_pred_df.drop(columns=['Outlet_Size', 'Item_Outlet_Sales']))

    out_train_pred_df['Outlet_Size'] = out_train_pred
    # out_test_pred = rf_model.predict(out_test_pred_df.drop(columns=['Outlet_Size']))
    # out_test_pred_df['Outlet_Size'] = out_test_pred

    df.dropna(inplace=True)
    # test_df.dropna(inplace=True)

    df = pd.concat([df, out_train_pred_df])
    # test_df = pd.concat([test_df, out_test_pred_df])
    return df


# Data Format (dictionary): {'ID':'Axsd34','Outlet_Size':'Medium',...}
# def createDataFrameUsingForm(data: dict) -> pd.DataFrame:
#     df = pd.DataFrame.from_dict(data,orient='index').T
#     return df

def train_model(train_df):

    X = train_df.drop(columns=['Item_Outlet_Sales'], axis=1)
    y = train_df['Item_Outlet_Sales']

    trainX, testX, trainY, testY = train_test_split(X, y, random_state=42, test_size=0.25)

    clf = Pipeline([('cat_reg', CatBoostRegressor(random_state=2, iterations=3000, learning_rate=0.002, depth=6, silent=True))])
    clf.fit(trainX, trainY)

    # print('Trining R2 Score: {}'.format(clf.score(trainX, trainY)))

    pred = clf.predict(testX)
    model_name = 'CatBoostRegressor'

    score_data = predictionResult(testY, pred, model_name)
    # logging training scores to file
    logToFile('../../other/logs/train_log.txt', score_data)
    dump(clf, '../../../models/model.pkl')

def predictionResult(testY,pred,model_name):

    model_acc_scores = {}
    # print('------------------Test Result---------------')
    # print('--------------------{}------------------'.format(model_name))
    score = r2_score(testY,pred)
    mae = mean_absolute_error(testY,pred)
    mse = mean_squared_error(testY,pred)
    rmse = np.sqrt(mse)
    model_acc_scores = {'r2_score':score,'mae_score':mae,'mse_score':mse,'rmse_score':rmse,'model_name':model_name}
    # print('R Squared Score is: {}'.format(score))
    # print('Mean Absolute Error is: {}'.format(mae))
    # print('Mean Squared Error is: {}'.format(mse))
    # print('Root Mean Squared Error is: {}'.format(rmse))
    return model_acc_scores


# Log train data to file

def logToFile(path: str, data: dict):
    # For logging training scores to file
    with open(path, 'a') as logfile:
        currTimestamp = datetime.datetime.now()
        logfile.write('------------{}---------------\n'.format(currTimestamp))
        logfile.write('Model Name: {} \n'.format(data['model_name']))
        logfile.write('R2 Score: {} \n'.format(data['r2_score']))
        logfile.write('Mean Abs Error: {} \n'.format(data['mae_score']))
        logfile.write('Mean Sq Error: {} \n'.format(data['mse_score']))
        logfile.write('Root Mean Sq Error: {} \n'.format(data['rmse_score']))
        logfile.write('-----------------------------------------------------\n\n')



