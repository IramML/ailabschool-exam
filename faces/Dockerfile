FROM python:3.7

WORKDIR /src/app
COPY model/ model/
COPY index.py .
COPY custom_objects.py .

RUN python3 -m pip install --upgrade pip

RUN pip install tensorflow keras pillow pydantic python-multipart
RUN pip install FastAPI uvicorn

EXPOSE 8500
CMD ["python3", "index.py"]%  