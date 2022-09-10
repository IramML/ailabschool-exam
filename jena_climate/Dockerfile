FROM python:3.7

WORKDIR /src/app
COPY model/ model/
COPY in/ in/
COPY index.py .
COPY utils.py .

RUN python3 -m pip install --upgrade pip

RUN pip install tensorflow==2 protobuf~=3.19.0 numpy==1.16.4 keras pandas h5py==2.10.0
RUN pip install FastAPI uvicorn


EXPOSE 8300
CMD ["python3", "index.py"]%  