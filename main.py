"""
This is an API Script for Fraud Detection.
"""

# Getting all importannt libraries
import uvicorn
from fastapi import FastAPI, Depends
from pydantic import create_model, BaseModel
from fastapi import Query
import numpy as np
import pickle
import pandas as pd

# Initializing the application
app = FastAPI()

# The Home Page
@app.get('/')
async def index():
    return {"note":"Lets start Fraud Prediction. Add an '/docs' extension to the url."}


# Loading the model
pickle_in = open("bestmodel.pkl","rb")
best_grid=pickle.load(pickle_in)


# Getting the features from features.txt
ft = []

# Generate inputs
class inputs(BaseModel):
    AC_1002_Issue_logT: int = 0
    Claim_Value_logT_bins: int = 0
    Region_North_West: int = 0
    Region_South: int = 0
    Purchased_from_Dealer: int = 0
    Purchased_from_Internet: int = 0
    Purchased_from_Manufacturer: int = 0
    Region_East: int = 0
    Region_North: int = 0
    Region_North_East: int = 0
    Region_South_East: int = 0
    Region_West: int = 0
    Area_Rural: int = 0
    Area_Urban: int = 0
    Consumer_profile_Business: int = 0
    Consumer_profile_Personal: int = 0
    Product_category_Entertainment: int = 0
    Product_category_Household: int = 0
    Product_type_AC: int = 0
    Product_type_TV: int = 0
    Purpose_Claim: int = 0
    Purpose_Complaint: int = 0


# Create a route
@app.post("/items/{fraud}")
async def fraudItems(params: inputs):
    '''
    This is the API Function we will be using to predict the Fraud
    '''

    # All the variables stored in the var is called as params and we are storing it as dict in params_dict
    params_dict = params.dict()

    # Since we just need the values to predict the fraud, lets get rid of keys.
    values = list(params_dict.values())
    prediction = best_grid.predict([values])
    probability = best_grid.predict_proba([values]).max()
    
    if(prediction[0]>0.5):
        prediction="Fraud"
    else:
        prediction="Not a fraud"

    return {
        'prediction': prediction, 'probability': probability

    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
