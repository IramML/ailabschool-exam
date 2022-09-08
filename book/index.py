from email import message
import uvicorn
from fastapi import FastAPI, Form
from pydantic import BaseModel
from tensorflow import keras 
import os
from autocomplete import AutocompleteUtils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = FastAPI()

model = None
utils = None

class PredictRB(BaseModel):
    image: str
    class Config:
        orm_mode = True

def load_local_model():
    global model
    model = keras.models.load_model('model/book.hdf5')

def load_utils():
    global utils
    utils = AutocompleteUtils()

@app.get("/predict/")
async def predict(text: str):
    text = utils.complete_text(model, text, n_charts=50, temperature=0.5)
    return {"code": 200, "message": "Request successfull", "text": text}

if __name__ == "__main__":
    load_local_model()
    load_utils()
    uvicorn.run(app, host='0.0.0.0', port=8400)