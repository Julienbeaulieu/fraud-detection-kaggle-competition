# [Kaggle competition: IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/overview) 

Can we detect fraud from customer transactions? Lessons from my first competition.

# 1. Motivation


After having spent a lot of time taking data science classes, I was eager to start practicing on a real dataset and to enter a Kaggle competition. I am thankful that I did because in the process, I learned a lot of things that aren't covered in those classes. Techniques like stratified cross validation, increasing the memory efficiency of my dataset, model stacking and blending, were all new to me.   

I also applied techniques learnt in Fastai's Intro to machine learning course which I'll comment on throughout the notebook. I highly recommend this course if you are learning like me. 

Even though my ranking was nothing impressive (top 45%), I now understand what it takes to create a state of the art kernel and have learned the tools to do so intelligently and efficiently. 

I am sharing my solution, methodology and a bunch of efficient helper functions as a way to anchor these learnings and for beginners who want to get better.

Topics covered in the notebook:
- Fastai helper functions to clean and numericalize a data set.
- A function to reduce the memory usage of your DataFrame.
- A methodology to quickly run a model to gain a better understanding of our dataset
- How Exploratory Data Analysis informs our feature engineering decisions
- Feature selection using LGBM's ` feature_importance` attribute.
- Crossvalidation for TimeSeriesSplit and StratifiedKFold with LGBM
- The code and helper functions for stacking and ensembling models
- Fastai tips and tricks throughout the notebook

# 2. About this dataset

In this competition we are predicting the probability that an online transaction is fraudulent, as denoted by the binary target `isFraud`. The data comes from [Vesta Corporation's](https://trustvesta.com/) real-world e-commerce transactions and contains a wide range of features from device type to product features.

The data is broken into two files: identity and transaction, which are joined by `TransactionID`.

> Note: Not all transactions have corresponding identity information.



**Evaluation**

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.
