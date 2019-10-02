# Motivation
After having spent a lot of time taking data science classes, I was eager to start practicing on a real dataset and to enter a Kaggle competition. I am thankful that I did because in the process, I learned a lot of things that aren't covered in those classes. Techniques like stratified cross validation, increasing the memory efficiency of my dataset, model stacking and blending, were all new to me.   

I also applied techniques learnt in Fastai's Intro to machine learning course which I'll comment on throughout the notebook. I highly recommend this course if you are learning like me. 

Even though my ranking was nothing impressive (top 60%), I now understand what it takes to create a state of the art kernel and have learned the tools to do so intelligently and efficiently. 

I am sharing my solution, methodology and a bunch of efficient helper functions as a way to anchor these learnings and for beginners who want to get better.

# About the data
In this competition we are predicting the probability that an online transaction is fraudulent, as denoted by the binary target `isFraud`. The data comes from [Vesta Corporation's](https://trustvesta.com/) real-world e-commerce transactions and contains a wide range of features from device type to product features.

The data is broken into two files: identity and transaction, which are joined by `TransactionID`.

> Note: Not all transactions have corresponding identity information.



**Evaluation**

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

# Methodology
**1. Quick cleaning and modeling**.
When starting off with a dataset with as many columns as this one (over 400), we'll want to quickly run it through an ensemble learner, forgoing any exploratory data analysis and feature engineering at the beginning. Once the model is fit to the data, we'll have a look at the features which are the most important using LGBM's feature_importances() method. 
This will allow us to concentrate our efforts on only the most important features instead of spending time looking at features with little to no predictive power. 

**2. Understand the data with EDA**.
Once we've filtered our columns, we'll look at the ones with the highest importance. The findings in this analysis will guide our feature engineering efforts in the next section. Some questions we'll want to answer:
- How are the top features related to our target variable? 
- What are their distributions like if we plot them with histograms and countplots?  
- What's their relationship with other important features? Do they seem to be related?
- Are there any features that we can split into multiple columns or simplify in any way?
- etc.

**3. Feature engineering**.
Once we understand our data, we can start creating new columns by splitting up current ones, transforming them to change their scale or looking at their mean, combining new ones to create interactions, and much more. 

**4. Train different models, fit them to the training data with cross validation, and perform model stacking and/or blending**.
The models I tested were RandomForests, XGBoost, and LightGBM. 
I tried several cross validation techniques such as Stratification and TimeSeriesSplit, neither of which beat my single model LGBM, but it was a great learning experience to code it. 
I discovered several powerful ensemble techniques which are used by top Kaggle contenders: stacking, blending, averaging our least correlated submissions, etc. I wasn't able to increase my performance much using these techniques, but again I learned a lot by trying. The performance wasn't great because I didn't create enough high performing differentiated models that I would stack together. After a little bit of testing, I realized that this would have consumed a lot of time, for maybe a slight increase in performance. My goal with this competition was to get an overview of how a competition works, not spend weeks fine tuning a few models. I am convinced that with more time I could have gotten better results. 

Here is the high-level summary of my different submissions:

- Base features with stock LGBM: 0.8934
- Add LGBM hyperparameter tuning: 0.9337 (+0.0403)
- Add feature selection: 0.9350 (+0.0013)
- Use TimeSeriesSplit crossvalidation: 0.9241 (-0.0109)
- Use Stratified crossvalidation: 0.9300 (-0.0050 vs top score)
- Stack 3 tree based models for level 1, LGBM for level 2: 0.8793 (-0.0557 vs top score) (base models weren't optimized which explains the poor performance)
- Use a weighted average on my top submissions: 0.9365 (+0.0015) - Best and final result. 