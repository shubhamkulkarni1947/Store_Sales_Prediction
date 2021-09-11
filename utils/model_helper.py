import pandas as pd


def removeDuplicateRows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df

# Data Format (dictionary): {'ID':'Axsd34','Outlet_Size':'Medium',...}
def createDataFrameUsingForm(data: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(data,orient='index').T
    return df

def fillNaN(df: pd.DataFrame) -> pd.DataFrame:
    #Create pkl for filling NaN of outletSize
    return df


