from flask import Flask, render_template, request, redirect
from models import db, SalesModel
import json
import logging


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)


@app.before_first_request
def create_table():
    db.create_all()


@app.route('/sales/create', methods=['GET', 'POST'])
def create():
    if request.method == 'GET':
        return {"message":"please hit post request"}
    if request.method == 'POST':
        data=request.get_json(force=True, silent=False, cache=True)
        logging.info(f"post request came with data {data}")
        sales = SalesModel(data['Item_Identifier'],data['Item_Weight'],data['Item_Fat_Content'],
                              data['Item_Visibility'],data['Item_Type'],data['Item_MRP'],
                              data['Outlet_Identifier'],data['Outlet_Establishment_Year'],data['Outlet_Size'],
                              data['Outlet_Location_Type'],data['Outlet_Type'],data['Item_Outlet_Sales'])
        db.session.add(sales)
        db.session.commit()

        return redirect('/sales/train')


@app.route('/sales/train')
def RetrieveList():
    Sales = SalesModel.query.all()
    return {"Sales":[SalesModel.json(sal) for sal in Sales]}



if __name__=='__main__':
    app.run(debug=True,host='localhost', port=5000)