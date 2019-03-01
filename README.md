[image1]: ./sentiment-model_picture.png "sentiment-model"
# Deploying-a-Sentiment-Analysis-Model
## Introduction
In this project you will construct a recurrent neural network or convolutional neural network for the purpose of determining the sentiment of a movie review using the IMDB data set. You will create this model using Amazon's SageMaker service. In addition, you will deploy your model and construct a simple web app which will interact with the deployed model.
Under construction

![sentiment-model][image1]

## Topics:
In this project, we cover several areas:
* Machine learning
* Natural language processing
* RNN (LSTM, GRU)
* Word embedding
* Pytorch
* AWS
* GPU computing
* End-to-end web application

The project can be extended to include the word2vec embedding (Skip Grams model, continuous-bag-of-words, negative sampling, etc): 


## Project Instructions
The deployment project which you will be working on is intended to be done using Amazon's SageMaker platform. In particular, it is assumed that you have a working notebook instance in which you can clone the deployment repository.

### Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/thomasxmeng/Deploying-a-Sentiment-Analysis-Model.git
		cd Deploying-a-Sentiment-Analysis-Model
	```

2. Make sure you have already installed the necessary Python packages according to the README in the program repository.
3. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.

### Libraries
The list below represents main libraries and its objects for the project.

* [Amazon SageMaker](https://ap-northeast-2.console.aws.amazon.com/sagemaker/home?region=ap-northeast-2#/landing). (Build, train, and deploy a model)
* [PyTorch](https://pytorch.org/) (LSTM classifier)

## Delete the Endpoint
Remember to always SHUT DOWN YOUR ENDPOINT if you are no longer using it. You are charged for the length of time that the endpoint is running so if you forget and leave it on you could end up with an unexpectedly large bill.
```
  predictor.delete_endpoint()
```

## Further improvements:
* A model which can give score and ratings to the level of postiveness or negativeness.
* Implement other RNN architectures. 
* We can improve the model by using larger Vocabs. 
* Use attention.
* Use word2vec pre-trained, negative sampling.
* Use A/B test to compare with different models and algorithms.
* The validation loss should be implemented too. 
* Use CNN 1D layer and pooling layer before or after the LSTM layers. (Combined model)
* Use attention techniques.
