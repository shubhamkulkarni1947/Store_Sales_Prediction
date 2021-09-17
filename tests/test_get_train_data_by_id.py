import json


def test_get_train_data_by_id_1(app, client):
    res = client.get('/user/train/1')
    assert res.status_code == 200
    expected = {"data": [{
        "Item_Fat_Content": "Regular",
        "Item_Identifier": "DRC01",
        "Item_MRP": 48.2692,
        "Item_Outlet_Sales": 443.4228,
        "Item_Type": "Soft Drinks",
        "Item_Visibility": 0.019278216,
        "Item_Weight": 5.92,
        "Outlet_Establishment_Year": 2009,
        "Outlet_Identifier": "OUT018",
        "Outlet_Location_Type": "Tier 3",
        "Outlet_Size": "Medium",
        "Outlet_Type": "Supermarket Type2",
        "id": 1
    }],
        "message": "Success",
        "status": True
    }
    assert expected == json.loads(res.get_data(as_text=True))

# Test case: for id which is not present in the DB

def test_get_train_data_by_id_2(app, client):
    res = client.get('/user/train/99999')
    assert res.status_code == 200
    expected = {
        "data": [],
        "message": "Success",
        "status": True
    }
    assert expected == json.loads(res.get_data(as_text=True))
