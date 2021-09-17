import json
from io import BytesIO


def test_predict_1(app, client):
    data = [{
        "Item_Identifier": "FDW58",
        "Item_Weight": 20.750,
        "Item_Fat_Content": "Low Fat",
        "Item_Visibility": 0.007565,
        "Item_Type": "Snack Foods",
        "Item_MRP": 107.8622,
        "Outlet_Identifier": "OUT049",
        "Outlet_Establishment_Year": 1999,
        "Outlet_Size": "Medium",
        "Outlet_Location_Type": "Tier 1",
        "Outlet_Type": "Supermarket Type1"
    },
        {
            "Item_Identifier": "FDB58",
            "Item_Weight": 10.5,
            "Item_Fat_Content": "Regular",
            "Item_Visibility": 0.013496,
            "Item_Type": "Snack Foods",
            "Item_MRP": 141.3154,
            "Outlet_Identifier": "OUT046",
            "Outlet_Establishment_Year": 1997,
            "Outlet_Size": "Small",
            "Outlet_Location_Type": "Tier 1",
            "Outlet_Type": "Supermarket Type1"
        }
    ]

    url = '/user/predict'
    res = client.post(url, data=json.dumps(data))
    assert res.status_code == 200
    response_data = json.loads(res.get_data(as_text=True))
    assert 'Item_Outlet_Sales' in response_data['data'][0]
    assert 'Item_Outlet_Sales' in response_data['data'][1]


# Test case for uploading csv file and predicting

def test_predict_2(app, client):
    data = {
        'file': (BytesIO(b'Item_Fat_Content,Item_Identifier,Item_MRP,Item_Type,Item_Visibility,Item_Weight,'
                         b'Outlet_Establishment_Year,Outlet_Identifier,Outlet_Location_Type,Outlet_Size,'
                         b'Outlet_Type\nLow Fat,FDW58,107.8622,Snack Foods,0.007565,20.75,1999,OUT049,Tier 1,Medium,'
                         b'Supermarket Type1\nRegular,FDB58,141.3154,Snack Foods,0.013496,10.5,1997,OUT046,Tier 1,'
                         b'Small,Supermarket Type1'), 'test_pred.csv')
    }

    url = '/user/predict'
    res = client.post(url, data=data)
    assert res.status_code == 200
    response_data = json.loads(res.get_data(as_text=True))
    assert 'Item_Outlet_Sales' in response_data['data'][0]
    assert 'Item_Outlet_Sales' in response_data['data'][1]


# Test Case: Typo in values of Item_Fat_Content

def test_predict_3(client):
    data = [{
        "Item_Identifier": "FDW58",
        "Item_Weight": 20.750,
        "Item_Fat_Content": "Low Fat",
        "Item_Visibility": 0.007565,
        "Item_Type": "Snack Foods",
        "Item_MRP": 107.8622,
        "Outlet_Identifier": "OUT049",
        "Outlet_Establishment_Year": 1999,
        "Outlet_Size": "Medium",
        "Outlet_Location_Type": "Tier 1",
        "Outlet_Type": "Supermarket Type1"
    },
        {
            "Item_Identifier": "FDB58",
            "Item_Weight": 10.5,
            "Item_Fat_Content": "Regul",
            "Item_Visibility": 0.013496,
            "Item_Type": "Snack Foods",
            "Item_MRP": 141.3154,
            "Outlet_Identifier": "OUT046",
            "Outlet_Establishment_Year": 1997,
            "Outlet_Size": "Small",
            "Outlet_Location_Type": "Tier 1",
            "Outlet_Type": "Supermarket Type1"
        }
    ]
    url = '/user/predict'
    res = client.post(url, data=json.dumps(data))
    assert res.status_code == 200
    response_data = json.loads(res.get_data(as_text=True))
    assert response_data['status'] == False
