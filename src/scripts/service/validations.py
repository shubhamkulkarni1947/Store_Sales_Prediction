cols = ["Item_Identifier", "Item_Weight", "Item_Fat_Content",
    "Item_Visibility", "Item_Type", "Item_MRP",
    "Outlet_Identifier", "Outlet_Establishment_Year",
    "Outlet_Size", "Outlet_Location_Type",
    "Outlet_Type", "Item_Outlet_Sales"]

def validate_prediction_request(data: list):
    isValid: bool = True
    for obj in data:
        for key in obj.keys():
            if (key in cols) and key != '':
                isValid = False
                break
        if not isValid:
            break
    return isValid
