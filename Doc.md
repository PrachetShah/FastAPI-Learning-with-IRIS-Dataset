## Installing FastAPI:

Installing FastAPI is the same as any other python module, but along with FastAPI you also need to install **uvicorn** to work as a server. You can install both of them using the following command:-

```
pip install fastapi uvicorn
```

## Testing Our API:

The above code defined all the path operation in the file that we’ll name as basic-app.py. Now to run this file we’ll open the terminal in our directory and write the following command:-

```
uvicorn basic-app:app --reload
```

Now the above command follows the following format:-

- basic-app refers to the name of the file we created our API in.
- app refers to the FastAPI instance we declared in the file.
- –reload tells to restart the server every time we reload.

## Interactive API docs:

Now to get the above result we had to manually call each endpoint but FastAPI comes with Interactive API docs which can access by adding **/docs** in your path. To access docs for our API we’ll go to **http://127.0.0.1:8000/docs**. Here you’ll get the following page where you can test the endpoints of your API by seeing the output they’ll give for the corresponding inputs if any. You should see the following page for our API.

## Deploying our ML Model:

**Building Our Model**:

For this tutorial, we are going to use GuassianNB as our model and iris dataset to train our model on. To build and train our model we use the following code:

```
basic-app.py file
```

## The Request Body:

The data sent from the client side to the API is called a **request** body. The data sent from API to the client is called a **response body**.
<br><br>
To define our request body we’ll use BaseModel ,in pydantic module, and define the format of the data we’ll send to the API. To define our request body, we’ll create a class that inherits BaseModel and define the features as the attributes of that class along with their type hints. What pydantic does is that it defines these type hints during runtime and generates an error when data is invalid. So let’s create our request_body class:-

```
from pydantic import BaseModel

class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float
```

## Testing our API:

To test our API we’ll be using Swagger UI now to access that you’ll just need to add /docs at the end of your path. So go to **http://127.0.0.1:8000/docs.**

**And as you can see we got our class as the response. And with that we have successfully deployed our ML model as an API using FastAPI.**
