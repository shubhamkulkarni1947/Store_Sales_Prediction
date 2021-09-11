import logging
import json

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
def clean_data(train_df,test_df):

# ######################### handle_missing_value ############################
    wt_miss_val = train_df['Item_Weight'].isna().sum()
    print('Total missing value for Item_Weight is {} out of {}'.format(wt_miss_val, len(train_df['Item_Weight'])))

    size_miss_val = train_df['Outlet_Size'].isna().sum()
    print('Total missing value for Outlet_Size is {} out of {}'.format(size_miss_val, len(train_df['Outlet_Size'])))

    # imputing Missing Value for Item_Weight with Mean
    train_df['Item_Weight'].fillna(train_df['Item_Weight'].mean(), inplace=True)
    test_df['Item_Weight'].fillna(test_df['Item_Weight'].mean(), inplace=True)

    #TODO applying tree based model to predict the value because null are larger in numbers
    # print(train_df['Outlet_Size'].value_counts())


    per_missing_visib_train = len(train_df[train_df['Item_Visibility'] == 0]) / len(train_df) * 100
    per_missing_visib_test = len(test_df[test_df['Item_Visibility'] == 0]) / len(test_df) * 100
    print('Percent of missing values in Train->Item_Visibility are: {}%'.format(round(per_missing_visib_train, 2)))
    print('Percent of missing values in Test->Item_Visibility are: {}%'.format(round(per_missing_visib_test, 2)))

    # Imputing Item_Visibility missinng values with Median
    train_df['Item_Visibility'].replace({0: train_df['Item_Visibility'].median()}, inplace=True)
    test_df['Item_Visibility'].replace({0: test_df['Item_Visibility'].median()}, inplace=True)


# ############################# Removing duplicates##############################################

    train_df = get_train_df()
    test_df = get_test_df()

    train_df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}, inplace=True)
    test_df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}, inplace=True)

    return {"train_df":train_df,"test_df":test_df}

def feature_encoding(train_df,test_df):

    train_df['Item_Type'].replace(
        {'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Breads': 'Baking Goods', 'Seafood': 'Meat'}, inplace=True)
    test_df['Item_Type'].replace(
        {'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Breads': 'Baking Goods', 'Seafood': 'Meat'}, inplace=True)

    # Binary Encoding
    train_df['Item_Fat_Content'].replace({'Low Fat': 0, 'Regular': 1}, inplace=True)
    test_df['Item_Fat_Content'].replace({'Low Fat': 0, 'Regular': 1}, inplace=True)

    nominal_features = ['Outlet_Type', 'Item_Type', 'Outlet_Identifier']
    prefixes = ['out_type', 'item_type', 'out_id']

    train_df = one_hot(train_df, nominal_features, prefixes)
    test_df = one_hot(test_df, nominal_features, prefixes)

    # Encoding Ordinal columns
    outlet_size_ord = {'Small': 0, 'Medium': 1, 'High': 2}
    out_loc_ord = {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}

    train_df = ord_enc(train_df, 'Outlet_Size', outlet_size_ord)
    test_df = ord_enc(test_df, 'Outlet_Size', outlet_size_ord)

    train_df = ord_enc(train_df, 'Outlet_Location_Type', out_loc_ord)
    test_df = ord_enc(test_df, 'Outlet_Location_Type', out_loc_ord)

    # # Deriving new column called Years_Since_Established from Establishment Year
    train_df['Years_Since_Established'] = train_df['Outlet_Establishment_Year'].apply(lambda x: 2021 - x)
    test_df['Years_Since_Established'] = test_df['Outlet_Establishment_Year'].apply(lambda x: 2021 - x)



    return {"train_df":train_df,"test_df":test_df}

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

def remove_irrelevant_columns(train_df,test_df):
    train_df.drop(columns=['Item_Identifier'], axis=1, inplace=True)  # ,'Outlet_Identifier'
    test_df.drop(columns=['Item_Identifier'], axis=1, inplace=True)
    return {"train_df":train_df,"test_df":test_df}


def predict_missing_values_Outlet_size(train_df,test_df):
    out_train_pred_df = train_df[train_df['Outlet_Size'].isna()]
    out_test_pred_df = test_df[test_df['Outlet_Size'].isna()]
    out_train_df = train_df[~train_df['Outlet_Size'].isna()]  # for training
    out_train_df.isna().sum()
    # out_train_df['Outlet_Size'] = out_train_df['Outlet_Size'].replace({'Small':0,'Medium':1,'High':2})
    # out_train_df.drop(columns=['Item_Identifier','Outlet_Identifier'],inplace=True)
    X = out_train_df.drop(columns=['Outlet_Size', 'Item_Outlet_Sales'])
    y = out_train_df['Outlet_Size']
    trainX, testX, trainY, testY = train_test_split(X, y, random_state=22, test_size=0.2)
    cat_model = RandomForestClassifier(random_state=2)
    cat_model.fit(trainX, trainY)

    pred = cat_model.predict(testX)

    out_train_pred = cat_model.predict(out_train_pred_df.drop(columns=['Outlet_Size', 'Item_Outlet_Sales']))

    out_train_pred_df['Outlet_Size'] = out_train_pred
    out_test_pred = cat_model.predict(out_test_pred_df.drop(columns=['Outlet_Size']))
    out_test_pred_df['Outlet_Size'] = out_test_pred

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    train_df = pd.concat([train_df, out_train_pred_df])
    test_df = pd.concat([test_df, out_test_pred_df])
    return {"train_df":train_df,"test_df":test_df}


# Data Format (dictionary): {'ID':'Axsd34','Outlet_Size':'Medium',...}
def createDataFrameUsingForm(data: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(data,orient='index').T
    return df

def train_model(train_df):

    X = train_df.drop(columns=['Item_Outlet_Sales'], axis=1)
    y = train_df['Item_Outlet_Sales']

    trainX, testX, trainY, testY = train_test_split(X, y, random_state=42, test_size=0.25)

    clf = Pipeline([('cat_reg', CatBoostRegressor(random_state=2, iterations=3000, learning_rate=0.002, depth=6, silent=True))])
    clf.fit(trainX, trainY)

    print('Trining R2 Score: {}'.format(clf.score(trainX, trainY)))

    pred = clf.predict(testX)
    model_name = 'CatBoostRegressor'

    predictionResult(testY, pred, model_name)
    dump(clf, '../../../models/model.pkl')

def predictionResult(testY,pred,model_name):

    model_acc_scores = {}
    print('------------------Test Result---------------')
    print('--------------------{}------------------'.format(model_name))
    score = r2_score(testY,pred)
    mae = mean_absolute_error(testY,pred)
    mse = mean_squared_error(testY,pred)
    rmse = np.sqrt(mse)
    scores_dict = {'R2 Score':score,'Mean Absolute Error':mae,'Mean Squared Error':mse,'Root Mean Squared Error':rmse}
    model_acc_scores[model_name] = scores_dict
    print('R Squared Score is: {}'.format(score))
    print('Mean Absolute Error is: {}'.format(mae))
    print('Mean Squared Error is: {}'.format(mse))
    print('Root Mean Squared Error is: {}'.format(rmse))
