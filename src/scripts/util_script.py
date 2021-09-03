import logging
import json
import pandas as pd

# to load the train data once at the time of database creation
def load_data():
    train_df = pd.read_csv('data/raw/Train.csv')

    #perform all step till complete deletion of unwanted column
