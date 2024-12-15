# Skin-Cancer-Classification

The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.


## Model Architecture:


## Data

https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000


## How to Run the App:

## Method1
•Run the app.py file

•Go to http://localhost:5000/ on your browser

•Use the Upload and button to browse and upload the image you want

•Hit submit to get the results.

## Method 2
•Depploy it to Azure Webapp or Heroku App through github repository

•Go to url generated after deployment on your browser

•Use the Upload and button to browse and upload the image you want

•Hit submit to get the results.


## Skin_Cancer_Detection.ipynb:
This is the Jupyter notebook used to define and train the model.

## app.py:
This is the flask app that needs to run in order to use the webapp

## skin_cancer_detection.py:
This contains the definition of the CNN model.

## model.keras:
Contains the weights of the best model.
