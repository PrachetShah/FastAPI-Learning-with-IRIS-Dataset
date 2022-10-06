from fastapi import FastAPI
import uvicorn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pydantic import BaseModel
 
# Creating FastAPI instance
app = FastAPI()
 
# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

class response_Out(BaseModel):
    output : str
 
# Loading Iris Dataset
iris = load_iris()
 
# Getting our Features and Targets
X = iris.data
Y = iris.target
 
# Creating and Fitting our Model
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
 
# Creating and Fitting our Model
clf = GaussianNB()
clf.fit(X_train,y_train)


# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to Initial Launch!'}


# http://localhost:8000/?sepal_length=4&sepal_width=3&petal_length=4&petal_width=5
# Creating an Endpoint to receive the data
# to make prediction on.
@app.get('/predict', response_model=response_Out)
async def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    '''
    Used to Classify Plant based on their features trained on iris dataset
    - Example URL:
    >>> http://localhost:8000/?sepal_length=4.3&sepal_width=3&petal_length=4&petal_width=5
    '''
    # Making the data in a form suitable for prediction
    test_data = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
    ]]
    print(test_data) 
    # Predicting the Class
    class_idx = clf.predict(test_data)[0]
     
    # Return the Result
    decision = {
        "output": iris.target_names[class_idx],
    }
    return decision

clf.fit(X,Y)

# Sending Accuracy Score of the Model
@app.get("/modelScore")
async def score():
  y_pred = clf.predict(X_test)
  return {"output": f"Accuracy of our Gaussian Model is: {accuracy_score(y_test, y_pred)*100}%"}

# Sending Recall, Precision, and F1-Score of The Model
@app.get("/statistics")
async def stats():
  y_pred = clf.predict(X_test)
  recall = recall_score(y_test, y_pred, average='micro')
  precision = precision_score(y_test, y_pred, average='micro')
  f1Score = f1_score(y_test, y_pred, average='micro')
  return {'output': f'Precision: {precision}    Recall: {recall}    F1 Score: {f1Score}'}

# To run it, 'uvicorn basic-app:app --reload'