# from flask_sqlalchemy import SQLAlchemy
import json
import decimal
# db = SQLAlchemy()
from json import JSONEncoder


class SalesModel():
    # class SalesModel(db.Model):
    # __tablename__ = "sales"
    # #by default auto generated column
    # id = db.Column(db.Integer, primary_key=True)
    # #removed unwanted data column
    # #Item_Identifier=db.Column(db.String(), primary_key=True)
    # Item_Weight=db.Column(db.Float())
    # Item_Fat_Content=db.Column(db.String())
    # Item_Visibility=db.Column(db.Float())
    # Item_Type=db.Column(db.String())
    # Item_MRP=db.Column(db.Float())
    # #Outlet_Identifier=db.Column(db.String())
    # Outlet_Establishment_Year=db.Column(db.Integer())
    # Outlet_Size=db.Column(db.String())
    # Outlet_Location_Type=db.Column(db.String())
    # Outlet_Type=db.Column(db.String())
    # Item_Outlet_Sales=db.Column(db.Float())

    def __init__(self, Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP,
                 Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type,
                 Item_Outlet_Sales):
        self.Item_Identifier = Item_Identifier
        self.Item_Weight = Item_Weight
        self.Item_Fat_Content = Item_Fat_Content
        self.Item_Visibility = Item_Visibility
        self.Item_Type = Item_Type
        self.Item_MRP = Item_MRP
        self.Outlet_Identifier = Outlet_Identifier
        self.Outlet_Establishment_Year = Outlet_Establishment_Year
        self.Outlet_Size = Outlet_Size
        self.Outlet_Location_Type = Outlet_Location_Type
        self.Outlet_Type = Outlet_Type
        self.Item_Outlet_Sales = Item_Outlet_Sales

    def __repr__(self):
        return f'Sales(Item_Identifier={self.Item_Identifier},' \
               f'Sales(Item_Weight={self.Item_Weight},' \
               f'Item_Fat_Content={self.Item_Fat_Content},' \
               f'Item_Visibility={self.Item_Visibility},' \
               f'Item_Type={self.Item_Type},' \
               f'Item_MRP={self.Item_MRP},' \
               f'Outlet_Identifier={self.Outlet_Identifier},' \
               f'Outlet_Establishment_Year={self.Outlet_Establishment_Year},' \
               f'Outlet_Size={self.Outlet_Size},' \
               f'Outlet_Location_Type={self.Outlet_Location_Type},' \
               f'Outlet_Type={self.Outlet_Type},' \
               f'Item_Outlet_Sales={self.Item_Outlet_Sales}'

    def json(self):
        return {
            'Item_Identifier': self.Item_Identifier,
            'Item_Weight': self.Item_Weight,
            'Item_Fat_Content': self.Item_Fat_Content,
            'Item_Visibility': self.Item_Visibility,
            'Item_Type': self.Item_Type,
            'Item_MRP': self.Item_MRP,
            'Outlet_Identifier': self.Outlet_Identifier,
            'Outlet_Establishment_Year': self.Outlet_Establishment_Year,
            'Outlet_Size': self.Outlet_Size,
            'Outlet_Location_Type': self.Outlet_Location_Type,
            'Outlet_Type': self.Outlet_Type,
            'Item_Outlet_Sales': self.Item_Outlet_Sales
        }


class SalesModelEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return o.__dict__
