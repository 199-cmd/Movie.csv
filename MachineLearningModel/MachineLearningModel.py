"""
To create a machine learning model
a) Formulate the question
b) Gather and clean the data
c) Visualise the data
d) Train the algorithm
e) Evaluate the result based on the requirements.
"""
#import the pandas modules for process the data
import pandas as pd
#import the dataframes library for align the data in row and column.
from pandas import DataFrame as df
#import matplotlib for 2d plotting or visualize the data
import matplotlib.pyplot as mp
#for train the model
from sklearn.linear_model import LinearRegression
#read the data in csv format
data = pd.read_csv("Movie.csv")
#print(data)

#read data in row and column form.
X = df(data, columns=['production_budget'])
Y = df(data, columns=['worldwide_gross'])
mp.scatter(X,Y)
#label for the chart
mp.xlabel('Production revenue in USD')
mp.ylabel('Worldwide gross in USD')
#Create a Object of LinearRegression
regressionObject = LinearRegression();
regressionObject.fit(X,Y)
#predict the outcomes
mp.plot(X, regressionObject.predict(X), color = 'green')
#showinig the result
mp.show()

