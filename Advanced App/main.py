# !pip install fastapi uvicorn python-multipart

from fastapi import FastAPI, File, UploadFile
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

"""### Data Pre-Processing"""

# Loading Dataset
credit = pd.read_csv('german_credit_data.csv')
credit.head()

credit['Risk'].unique()

credit['Risk'] = credit['Risk'].replace('good', 1)
credit['Risk'] = credit['Risk'].replace('bad', 0)

credit['Sex'] = credit['Sex'].replace('male', 1)
credit['Sex'] = credit['Sex'].replace('female', 0)

data = credit[['Sex', 'Job', 'Credit amount', 'Duration', 'Risk']]
data

# Getting our Features and Targets
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# Creating and Fitting our Model
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
 
# Creating and Fitting our Model
clf = GaussianNB()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)*100

"""### Creating FastAPI Endpoints"""

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to Initial Launch!'}

# http://localhost:8000/?sex=0&job=3&credit=4000&duration=10
# Creating an Endpoint to receive the data
# to make prediction on.
@app.get('/predict')
async def predict(sex: float, job: float, credit: float, duration: float):
    '''
    Used to Classify Plant based on their features trained on iris dataset
    - Example URL:
    >>> http://localhost:8000/?sex=0&job=3&credit=400&duration=10
    '''
    # Making the data in a form suitable for prediction
    test_data = [[
            sex,
            job,
            credit,
            duration
    ]]
    # Predicting the Class
    class_idx = clf.predict(test_data)
    risk = class_idx[0]
     
    # Return the Result
    decision = {
        "input": [sex, job, credit, duration],
        "risk": int(risk),
    }
    return decision


# Getting Accuracy Score of the Model
@app.get("/accuracy")
async def score():
  y_pred = clf.predict(X_test)
  return {"output": f"Accuracy of our Gaussian Model is: {accuracy_score(y_test, y_pred)*100}%"}

# Sending Recall, Precision, and F1-Score of The Model
@app.get("/statistics")
async def stats():
  y_pred = clf.predict(X_test)
  recall = recall_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  f1Score = f1_score(y_test, y_pred)
  return {'output': f'Precision: {precision}    Recall: {recall}    F1 Score: {f1Score}'}

# Log Model Performance
from datetime import datetime

@app.get("/logger")
async def log():
  now = datetime.now()
  current_time = now.strftime("%H:%M:%S")
  with open("logging.txt", mode="a") as file:
    file.write(f'Accuracy of Model at {current_time} is {accuracy_score(y_test, y_pred)*100}\n')
  return {'Output': 'File Written Successfully'}

@app.post("/files")
def create_upload_file(file: UploadFile=File(...)):
    '''
    This function takes in csv file for training, trains the model on new file 
    and logs the accuracy of new model in to logging.txt file
    '''
    # contents = await file.read()
    credit = pd.read_csv(file.file)
    credit['Risk'].unique()

    credit['Risk'] = credit['Risk'].replace('good', 1)
    credit['Risk'] = credit['Risk'].replace('bad', 0)

    credit['Sex'] = credit['Sex'].replace('male', 1)
    credit['Sex'] = credit['Sex'].replace('female', 0)

    data = credit[['Sex', 'Job', 'Credit amount', 'Duration', 'Risk']]
    data

    # Getting our Features and Targets
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    # Creating and Fitting our Model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    # Creating and Fitting our Model
    clf = GaussianNB()
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    file.file.close()
    
    # Logging
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open("logging.txt", mode="a") as log:
        log.write(f'Accuracy of Model at {current_time} is {accuracy_score(y_test, y_pred)*100}\n')
    
    return {'filename': file.filename, 'Accuracy':f'{accuracy_score(y_test, y_pred)}'}

# To run it, 'uvicorn basic-app:app --reload'