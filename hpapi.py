from flask import Flask
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/')
def home():
    return "<p>Enter Values In URL!</p>"

@app.route("/<int:a>/<int:b>/<int:c>/<int:d>/<int:e>/<int:f>/<int:g>/<int:h>/<int:i>/<int:j>/<int:k>/<int:l>/<int:m>/<int:n>/<int:o>/<int:p>")
def predict(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p):
    paris=pd.read_csv('ParisHousing.csv')
    X = paris.drop('price', axis = 1)
    y = paris['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.3)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_test_prediction=regr.predict(X_test)
    y_test_prediction= y_test_prediction * 0.00001
    y_test=y_test * 0.00001
    #test_data_accuracy=accuracy_score(np.round(y_test_prediction),np.round(y_test))
    # a=int(input("SquareMeters:"))
    # b=int(input("NumberOfRooms:"))
    # c=int(input("HasYard:"))
    # d=int(input("HasPool:"))
    # e=int(input("Floors:"))
    # f=int(input("CityCode:"))
    # g=int(input("CityPartRange:"))
    # h=int(input("NumPrevOwners:"))
    # i=int(input("Made:"))
    # j=int(input("IsNewBuilt:"))
    # k=int(input("HasStormProtector:"))
    # l=int(input("Basement:"))
    # m=int(input("Attic:"))
    # n=int(input("Garage:"))
    # o=int(input("HasStorageRoom:"))
    # p=int(input("HasGuestRoom:"))
    input_data=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)
    arr= np.asarray(input_data).reshape(1,-1)
    prediction=regr.predict(arr)
    return f"The value of house should be approx:${prediction}"



