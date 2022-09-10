import uvicorn
from fastapi import FastAPI, Form
from tensorflow import keras 
import tensorflow as tf
import os
from utils import Utils
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = FastAPI()

project_path = os.path.dirname(os.path.realpath(__file__))

model = None
utils = None

def load_local_model():
    global model
    model = keras.models.load_model('{}/model/model.h5'.format(project_path), compile=False)

def load_utils():
    global utils
    utils = Utils()

@app.get("/predict/")
async def predict(p: float, t_deg: float, t_spot: float, t_dew: float, rh: float, vp_max: float, vp_act: float, vp_def: float, sh: float, H2OC: float, rho: float, wv: float, max_wv: float, wd: float, day: int, year: int):
    cos_day, sin_day = utils.cos_sin_day(day)
    cos_year, sin_year = utils.cos_sin_year(year)
    data = [p, t_deg, t_spot, t_dew, rh, vp_max, vp_act, vp_def, sh, H2OC, rho, wv, max_wv, wd, sin_day, cos_day, sin_year, cos_year] 

    data = [utils.normalize(data[i], i) for i in range(len(data))]
    input_data = tf.convert_to_tensor(np.array([[data]], dtype='f'), dtype=tf.float32)

    temperature_normalized = model(input_data)
    temperature_normal_value = utils.normalized_to_normal_value(temperature_normalized[0,0].numpy()[0], 1)
    return {"code": 200, "message": "Request successfull", "temperature": temperature_normal_value}

if __name__ == "__main__":
    load_local_model()
    load_utils()
    uvicorn.run(app, host='0.0.0.0', port=8300)