# Jena Climate

## How to build the project

### Generate the model
1. To generate and train the model you should use the Examen_R3 notebook.
2.  The only parameters you just need to are in the second cell of the notebook. The `project_path` and the third line where is changing of directory. You will replace them and you will put the path of the directory where is located your notebook. 
3. That's it, now just run all the cells of code

The notebook will create two folders, `in` and `out`, inside the out folder you will find the best model that was trained.

### How to run the fastapi server
1. Create a directory with the name `model`
2. The model that is inside of `out` folder, you will move it to the `model` directory you just created 
3. If is needed, edit the `index.py` to use the correct name and path of the model, in the line 20
4. Install with pip the dependencies: tensorflow, numpy, keras pandas, FastAPI and uvicorn
5. Run `Python3 index.py`
6. The server will be up in the port 8300, the model will be available to serve in the `/predict` endpoint
7. You can create a docker image of this backend using the Dockerfile that is on the project