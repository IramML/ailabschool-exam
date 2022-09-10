# AI Lab School Final Exam

## About the project

This repository is my solutions of the final exam of AI Lab School.
These are three projects in which deep learning was used to solve 3 problems.

### Face recognition
In this project I builded a model that uses face recognition to classify faces. The model was trained using a dataset of thermal faces in different angles. I used convolutional neuronal networks and a feed foward network to solve this problem. 

[Docker Hub](https://hub.docker.com/repository/docker/iramml/faces-tensorflow)

### Text autocomplete
This model is used to autocomplete text. The model was trained using a Nietzsche book. To solve this problem I used recurrent neural networks (GRU and LSTM). It also has it's backend to deploy the model and a dockerfile to create an image to deploy the server.

[Docker Hub](https://hub.docker.com/repository/docker/iramml/autocomplete-book)

### Climate forecast
This project was builded to predict the climate. The model was trained using the jena climate dataset and the model is made of recurrent neuronal networks (LSTM).

[Docker Hub](https://hub.docker.com/repository/docker/iramml/jena-climate)

## Getting Started

### Requirements
- [Python](https://www.python.org/)
- [Pip](https://pip.pypa.io/en/stable/installation/)
- [Jupyter](https://jupyter.org/) or [Google Colab](https://colab.research.google.com/)

### How to build the projects
- [Faces README](faces/README.md)
- [Book README](book/README.md)
- [Climate README](jena_climate/README.md)
