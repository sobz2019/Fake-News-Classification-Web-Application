#  Fake News Classification Web Application using Python

## Project Description

The goal of this project is to create a graphical user interface (GUI) web application that could predict whether a news article is fake or not, by using trained models that have been trained over open-source datasets with the utilisation of Natural Language Processing (NLP) concepts based on the article text content.

## Proposed Algorithms and Pretrained Models

### Machine Learning Algorithms

* Random Forest
* SVM
* KNN
* XgBoost
* Naive Bayes
* Logistic Regression


### Deep Learning Algorithms

* 1D CNN
* LSTM
* Bi-LSTM

### Pretrained Models

* Word2Vec = GoogleNews-vectors-negative300.bin
* Glove = glove.6B.300d.txt
* BERT = bert_en_uncased_L-12_H-768_A-12/1


## Datasets
* FakeNewsNet
* ISOT
* Fakeddit

## Getting Started

### Prerequisites

This section describes the software and libraries which need to install in the server as prerequisites for the experiment.

* Softwares

    * Python 3.7
    * Flask 2.1.2
    * Anaconda Navigator

* Python libraries

    The list of the required libraries present in the file named `requirement.txt` . You can install these libraries with pip command as below:

    `pip install -r requirements.txt`
    

### Folder Structure Description

* Documents

    This project directory  includes the architectural and design related files for the fake news automatic detection using machine learning and deep learning algorithms.

* Source Codes

    There are mainly 3 folders available in this directory 

    * Experimental Codes

       There are 3 subfolders within this directory, one for each dataset, which contain all the coding notebooks used to develop the model for automatic fake news detection over the respective datasets.

       * FakeNewsNet
       * ISOT
       * Fakeddit
       
       Along with the coding notebooks in each folder dataset, 4 more subfolders present and the purpose for these 4 folders is described as follows:

       * Embeddings -  Folder to store the downloaded pretrained embedding models 
       * input_dataset - Folder to store the donwloaded datasets 
       * outputs - Folder to store the trained model and its metric details
       * Updated - Folder to store the filtered data of each dataset 
     
       For each dataset, we have experimented model development based on 4 different feature extraction techniques and created notebooks seperatly based on this.


    * General Codes

        This folder contains multiple coding notebooks which describes the general steps and scripts which we used in this experiment.These notebooks are arranged based on different phases of the machine learning model development done for the automatic fake news detection.

    * Model Deployment

        This folder represents the model deployment codes associated with the GUI web application developed for this project



### Model Deployment Steps

- Clone this repository.
- Open Command Prompt from the working directory.
- Run `pip install -r requirements.txt`
- Open the project from any IDE
- Run `Fake_News_Det.py`
- Go to the URL `http://127.0.0.1:5100/`




