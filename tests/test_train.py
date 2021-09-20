import json


def test_train_1(client):
    data = [
        {
            "Item_Fat_Content": "Low Fat",
            "Item_Identifier": "FDW58",
            "Item_MRP": 107.8622,
            "Item_Outlet_Sales": 1713.405,
            "Item_Type": "Snack Foods",
            "Item_Visibility": 0.007565,
            "Item_Weight": 20.75,
            "Outlet_Establishment_Year": 1999,
            "Outlet_Identifier": "OUT049",
            "Outlet_Location_Type": "Tier 1",
            "Outlet_Size": "Medium",
            "Outlet_Type": "Supermarket Type1"
        },
        {
            "Item_Fat_Content": "Regular",
            "Item_Identifier": "FDB58",
            "Item_MRP": 141.3154,
            "Item_Outlet_Sales": 2201.606,
            "Item_Type": "Snack Foods",
            "Item_Visibility": 0.013496,
            "Item_Weight": 10.5,
            "Outlet_Establishment_Year": 1997,
            "Outlet_Identifier": "OUT046",
            "Outlet_Location_Type": "Tier 1",
            "Outlet_Size": "Small",
            "Outlet_Type": "Supermarket Type1"
        }
    ]

    url = '/user/train'
    res = client.post(url, data=json.dumps(data))
    assert res.status_code == 200
    response_data = json.loads(res.get_data(as_text=True))
    assert response_data['status'] is True
