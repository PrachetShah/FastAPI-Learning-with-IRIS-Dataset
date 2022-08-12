from fastapi import FastAPI, Request
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
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
clf = GaussianNB()
clf.fit(X,Y)


# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to Initial Launch!'}


# http://localhost:8000/?sepal_length=4&sepal_width=3&petal_length=4&petal_width=5
# Creating an Endpoint to receive the data
# to make prediction on.
# @app.get('/predict', response_model=response_Out)
# async def predict(data : request_body):
#     # Making the data in a form suitable for prediction
#     print(data)
#     test_data = [[
#             data.sepal_length,
#             data.sepal_width,
#             data.petal_length,
#             data.petal_width
#     ]]
#     print(test_data)
     
#     # Predicting the Class
#     class_idx = clf.predict(test_data)[0]
     
#     # Return the Result
#     decision = {
#         "output": iris.target_names[class_idx],
#     }
#     return decision

@app.get('/predict', response_model=response_Out)
async def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
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

# To run it, 'uvicorn basic-app:app --reload'