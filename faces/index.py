from email import message
import uvicorn
from fastapi import FastAPI, UploadFile, Form, File
from pydantic import BaseModel
from tensorflow import keras 
import os
import io
from PIL import Image, ImageOps
from custom_objects import recall_metric, precision_metric, f1_metric

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np

app = FastAPI()

project_path = os.path.dirname(os.path.realpath(__file__))

model = None

class PredictRB(BaseModel):
    image: str
    class Config:
        orm_mode = True

def load_local_model():
    global model
    model = keras.models.load_model('{}/model/faces.h5'.format(project_path), custom_objects={"recall_metric": recall_metric, "precision_metric": precision_metric, "f1_metric": f1_metric })

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    image_shape = (512,512)
    image_content = await image.read()
    image = Image.open(io.BytesIO(image_content)).convert('LA')
    image = image.resize(image_shape, Image.ANTIALIAS)
    image = ImageOps.grayscale(image)
    image = keras.preprocessing.image.img_to_array(image)
    image = np.reshape(image, image_shape)

    labels = ["Face01", "Face02", "Face03", "Face04", "Face07", "Face08", "Face09", "Face10", "Face11", "Face12", "Face13", "Face14", "Face15", "Face16", "Face17", "Face18", "Face19", "Face30"]

    predict = model.predict(np.array([image]))

    label = labels[np.argmax(predict[0])]
    return {"code": 200, "message": "Request successfull", "face": label}

if __name__ == "__main__":
    load_local_model()
    uvicorn.run(app, host='0.0.0.0', port=8500)