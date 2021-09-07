#import statements
import src.scripts.dao.database_operations as dao




# function for training the model
def train_model():
    #TODO : train the model follow complete process
    pass

# function for predicting the value
def predict_sales_csv(test_csv_filepath):
    # TODO :load test file as df
    #predict the csv file data and return it as json
    pass

def predict_sales(data):
    #TODO : json data with sales data parameter
    #extract and drop feature
    #predict the sales and return
    pass

# other supporting function


def load_train_csv_to_db(filepath):
    dao.load_training_csv_data(filepath)

# validate the data
def upload_a_train_data_to_db(data):
    dao.insert_a_train_data(data)

def get_train_data_from_db():
    data=dao.get_train_data()
    return data