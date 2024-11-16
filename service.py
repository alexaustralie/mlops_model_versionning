import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import random

app = FastAPI()

mlflow.set_tracking_uri(uri="http://192.168.122.1:8080")

# Load both current and next model (at startup both will be the same)
model_name = "tracking-quickstart"
current_model_version = 1  # Initial model version
next_model_version = 1  # Same version initially
current_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{current_model_version}")
next_model = current_model  # Initially both are the same

class PredictRequest(BaseModel):
    data: list

@app.post("/predict")
async def predict(request: PredictRequest):
    input_data = pd.DataFrame(request.data)
    try:
        p = 0.7  
        use_current = random.random() < p  

        if use_current:
            predictions = current_model.predict(input_data)
        else:
            predictions = next_model.predict(input_data)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint to update the next model
class UpdateModelRequest(BaseModel):
    version: int

@app.post("/update-model")
async def update_model(request: UpdateModelRequest):
    global next_model
    try:
        next_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{request.version}")
        return {"message": f"Next model updated to version {request.version}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint to accept the next model as current
@app.post("/accept-next-model")
async def accept_next_model():
    global current_model, next_model
    try:
        current_model = next_model
        return {"message": "Next model is now the current model."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
