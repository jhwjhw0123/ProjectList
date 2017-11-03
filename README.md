# ProjectList<br />
This is a project listing my previous projects. This will be keeping update as I am carrying out more projects.<br />
Last Modified: Nov/3/2017 . <br />
**1.	RNN for MNIST label prediction and data in-painting.** <br />
Key Words: Deep Learning, Recurrent Neural Network, MNIST <br />
This is a coursework of the course ‘Advanced Topics in Machine Learning’ delivered by Google DeepMind. In this project, I developed two kinds of RNNs, one for image label prediction and the other for the image in-painting. The label prediction task split the MNIST images into pixel-by-pixel sequence, and hence the training task is non-trivial. The second RNN network learns the long-term dependency of the image and to fill missing pixels if some of the pixels are corrupted. Both LSTM and GRU cells are tested in the project.<br />
The source codes of this project are available on GitHub:<br />
https://github.com/jhwjhw0123/RNN_sequence_image_prediction_and_in-painting. <br />
**2.	Deep Reinforcement Learning for OpenAI Gym Games.** <br />
Key Words: Deep Reinforcement Learning, Neural Network Q-learning, DQN, Double-DQN <br />
This is also a coursework of the course ‘Advanced Topics in Machine Learning’ delivered by Google DeepMind. In this project, different agents based on neural networks are trained to play Gym games in the environment provided by openAI. The first part is designed for CartPole game and then the second part is deployed for Pong, Ms. Pancman and Boxing respectively. Batch Q-learning, online Q-learning, DQN and double-DQN are implemented respectively, and the result had shown a significant improvement for the games’ performance comparing to random policies.<br />
The source codes of this project are available on GitHub:<br />
https://github.com/jhwjhw0123/Deep_Reinforcement_Atari_games_ATML <br />
**3.	Master Project: Embedded System Neural Network Library and Related Algorithms** <br />
Key Words: Embedded Deep Learning, C++, Optimization, Cross-compiling, Pruning, Quantization, Sparse Programming <br />
This is an integrated project submitted as my Master’s project, which covers the area of Deep Learning, Embedded Machine Learning, C++ programming techniques, and related modifying algorithms like Quantization and Pruning. A neural network library is produced from rudiments with the help of a matrix multiplication package, and many useful features are added to the networks for embedded usage. To modify some properties of the network, Machine Learning-based Quantization and Pruning techniques are applied and a novel algorithm to perform quantization is designed. <br />
The source codes of this project are not yet available because it is still under scrutiny of my supervisor. <br />
**4.	Undergraduate Research Project: The Study of Intelligent Car Based on Artificial Intelligent and Fuzzy Theorem** <br />
Key Words: Embedded System, C Programming Language, Boosting Method, Fuzzy Logic, Fuzzy Network <br />
This is my undergraduate project which I finished one year ago. The study was focused on the using Machine Learning methods and Fuzzy Logic to improve the performance of a standard intelligent car. Adaboost method was used to discriminate the running mode of the car based on the distance from the obstruction and the current speed, and a fuzzy network is employed to improve the performance when the car is turning. The project received an honour of Distinction, which only granted to 18 out of more than 300 graduating students. <br />
The source codes of this project are available on GitHub: <br />
https://github.com/jhwjhw0123/Undergraduate_Project_Intelligent_Car. <br />
**5.	UCL Data Science Student Challenge – Financial Instrument Prediction** <br />
Key Words: xgBoost, HSIC feature selection <br />
This project is for the UCL student Data Challenge hosted by UCL department of Computer Science, Microsoft and Bloomberg. Our task was to predict the increments or decrements in the next 20 days for our selected predictable financial instruments. This is a teamwork with other 3 teammates, and my work is to develop a feature selection method base on HSIC Lasso (paper: https://arxiv.org/pdf/1202.0515.pdf) to recognize the noise features and improve the prediction accuracy. I also took charge part of the work of developing the Machine Learning prediction model based on xgBoost. We finally achieved an accuracy of 68.75% for the test dataset. <br />
The codes of this project are available on GitHub: https://github.com/DSSCHack2017/UCL-24hours-data-science-challenge <br />
Notice that the HSIC program in this project is a pristine version. The optimization is not enough stable for large-scale dataset and many features are not covered. The advanced-developed version could be found in separate GitHub page (Python version): <br />
https://github.com/jhwjhw0123/HSIC_Lasso_Python_version <br />
**6.	Mixture of Gaussian Time Series Data Clustering** <br />
Key Words: Time Series Data, Novel Algorithm Implementation, Mixture of Gaussian Clustering <br />
This project implemented the algorithm based on paper ‘Gaussian Mixture Models for Time Series Modelling, Forecasting, and Interpolation’, which designs a special Mixture of Gaussian(MoG) model for time series clustering. Comparing to the standard MoG model, this new method modifies the original one by taking time series covariance into consideration. Both AIC and BIC methods for choosing numbers of Gaussians are implemented in the program, and the method could be potentially used in detecting underlying patterns and clustering time series. This would be especially useful for some time series that might have some crucial latent patterns, such like stock price time series. <br />
The codes of this program are available on GitHub:<br />
https://github.com/jhwjhw0123/Time_Series_MoG_Clustering <br />
**7.	Kaggle: House Price Prediction, feature selection part** <br />
Key Words: HSIC feature selection, MATLAB <br />
This project originated from the Kaggle competition of House Price Prediction, which demands us to build a regression model to predict the house price in a certain area. The biggest challenge of this project is the noisy features, which could adversely affect the prediction result. Here I again used HSIC Lasso to select the features and decrease the regression loss.
The task is finished by using MATLAB and this version of HSIC Lasso program had already been provided by the author of original paper. Nevertheless, in the original version the author only provided Dual Augmented Lagrange method for the optimization in the program. To explore optimization methods for the method, and to finish a project of the course ‘Numerical Optimisation’, I designed various numerical optimization methods, including Proximal Method, FISTA method, Proximal Newton, ADMM method and Dual Augmented Lagrange method, based on the HSIC skeleton and Kaggle House Price dataset. The performances of different methods are compared and analysed under various conditions. <br />
The codes of this project are available on <br />
GitHub: https://github.com/jhwjhw0123/HSIC_Lasso_with_optimization . <br />
However, according to the terms and conditions issued by Kaggle, the data would not be available for test.
**8.	CRNN for Stock Price Analysis from Candlestick chart** <br />
Key Words: CRNN, Image Recognition, Stock Price Analysis <br />
This is a project for a competition which aims to use the Candlestick charts to predict the price of the stock market in China. The standard approach is to use deep CNN to predict the result. To capture the time-dependency of the Candlestick chart, I implement a CRNN model inspired by the paper ‘An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition’ (https://arxiv.org/pdf/1507.05717.pdf). The method could achieve a satisfying result with a relatively small size of network.<br />
The codes of this project are not yet available since the competition is still going on. The codes might be released soon after the competition.<br />
**9.	Story Sentence Sort with RNN** <br />
Key Words: RNN, Natural Language Processing, Story Sorting<br />
This project is a coursework of the course ‘Statistical Natural Language Processing’ delivered by UCL. This is a group work, and in the task, we are demanded to design a Recurrent Neural Network with vector representations of words to get a satisfying sorting result. Our final sorting accuracy is 55.6% -- although it is not very high, considering the difficulty of the task, the result is still satisfying.<br />
The codes of this project (in Jupyter Notebook) are available on GitHub: https://github.com/jhwjhw0123/Story_Sort_NLP . <br />
**10.	Machine Vision Apple Recognition with Mixture of Gaussian** <br />
Key Words: Machine Vision, Mixture of Gaussian Model, MLE, E-M algorithm <br />
This project is a coursework of the course ‘Machine Vision’ delivered by UCL. The task demanded me to classify apple/non-apple (the fruit!) on the image according to the RGB values. I developed the multi-variate Mixture of Gaussian Model with E-M MLE optimization scheme by my own (without package), and the test result is satisfying. <br />
The codes of this project are available on GitHub: https://github.com/jhwjhw0123/Machine_Vision_Mixture_of_Gaussian .<br />
