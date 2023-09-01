
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

paris=pd.read_csv('D:\streamlit\ParisHousing.csv')

X = paris.drop('price', axis = 1)
y = paris['price']



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.3)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

regr = LinearRegression()
regr.fit(X_train, y_train)
y_test_prediction=regr.predict(X_test)

y_test_prediction= y_test_prediction * 0.00001
y_test=y_test * 0.00001

test_data_accuracy=accuracy_score(np.round(y_test_prediction),np.round(y_test))


a=int(input("SquareMeters:"))
b=int(input("NumberOfRooms:"))
c=int(input("HasYard:"))
d=int(input("HasPool:"))
e=int(input("Floors:"))
f=int(input("CityCode:"))
g=int(input("CityPartRange:"))
h=int(input("NumPrevOwners:"))
i=int(input("Made:"))
j=int(input("IsNewBuilt:"))
k=int(input("HasStormProtector:"))
l=int(input("Basement:"))
m=int(input("Attic:"))
n=int(input("Garage:"))
o=int(input("HasStorageRoom:"))
p=int(input("HasGuestRoom:"))
input_data=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)
arr= np.asarray(input_data).reshape(1,-1)
prediction=regr.predict(arr)
print("The value of house should be approx $",prediction)

pickle.dump(regr, open('model.pkl','wb'))

