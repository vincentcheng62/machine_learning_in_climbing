#!/usr/bin/env python
# coding: utf-8

# # Correlation and feature importance
# 
# Ok, after looking at the general trends on the data, as well as lookin at proof that different subgroups have very different behaviors, it is time to figure out how variables correlate with each other, as well as finding which are the variables that better describe the variable we are interested in.
# 
# In this case, climbers will probably want to learn which variables correlate with their climbing performance, so we will use some analytic tools to find which variables are the best at predicting climbing performance. And being good at predicting something, usually means that modifying that would modify the variable of interest as you desire (become better climbers in this example).

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV


# In[2]:


fullData = pd.read_csv("filteredData.csv").replace(np.nan,0)
fullData.describe()


# In[3]:


normData = pd.read_csv("normData.csv").replace(np.nan,0)
normData.describe()


# In[4]:


data = normData.copy()
data.iloc[:,-17:] = fullData.iloc[:,-17:]
data.describe()


# Ok, as you can see, we setup the data in a way that all the independent variables (climbing performance / grades) are in their natural form, while all the independent variables (the remaining variables) are in their normalized form. We do this because we want the variables to be able to predict the true value of the "output" variable, but we do not want the learning models to give more importance to some variable over others just because they operate on a greater magnitude.
# 
# First, lets start by getting the pearson correlation of the entire matrix and quickly analyze it.

# In[5]:


corrDF = data.corr().replace(np.nan,0)
corrDF.tail(17)


# In[6]:


absCorr = corrDF.abs()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(ax=ax, data=(1-absCorr), vmin=0, vmax=1)


# Ok, correlation is a measurement that tells us how related variables are. The higher the magnitude of the correlation, the more related the variables are. In the previous graph, since the plot is 1-corr instead of corr, the closer to 0 (black) the better. The reason why we can see a black diagonal is because every variable es completely correlated to itself.
# 
# However, just as in the previous notebook, it is hard to visualize anything if we show all the variables together, so lets try to focus again. This time, instead of manually selecting variables like in the previous notebook where we used different views, we can use the correlation itself and simply filter the best "n" variables out of those.
# 
# The correlation, specifically to "S_AVG" is the following.

# In[7]:


sortCorr = corrDF.sort_values(by="S_AVG",ascending=True).T.sort_values(by="S_AVG",ascending=True)
goal = sortCorr["S_AVG"]
goal.plot.barh(title="Correlations to S_AVG",figsize=(15, 10))


# In[8]:


sortCorr2 = corrDF.abs().sort_values(by="S_AVG",ascending=True).T.sort_values(by="S_AVG",ascending=True)
goal2 = sortCorr2["S_AVG"]
goal2.plot.barh(title="ABS Correlations to S_AVG",figsize=(10, 10))


# It is important to see them both as absolutes and as their signed value. The reason for this is because a variable with a correlation of -0.2 gives us more information than one with a correlation of 0.1, but we must be aware that the effect is the inverse. Negative correlation means, the higher the value of this variable, the lower the goal variable.
# 
# We can also manually filter other "goal" variables, since we would not be able to modify them in the first place so we do not care about them.

# In[9]:


removeVars = ["B_INFLASH","B_INREDP","B_OUTFLASH","B_OUTREDP","B_FLASHCONF",
              "B_SENDCOF","B_AVG","B_INPOTENTIAL","B_OUTPOTENTIAL",
              "S_FLASH","S_REDP","S_FLASHCONF", "S_FINISHCONF",
              "S_FPOTENTIAL","S_MPOTENTIAL","S_INCONSISTENCY"]
goal2.drop(removeVars, inplace=True)
goal2 = goal2[-40:-1]
print(len(goal2))
goal2.plot.barh(title="ABS Correlations to S_AVG",figsize=(10, 10))


# OK so how to interpret correlation? Easy, it basically means that knowing information about that variable lets you better predict the performance. In this case the things which one could modify and which have a higher impact in climbers' performance, in order, are:
# -  +How many days a week they climb (including indoors)
# -  +If they create their own training plan or something they read (-or if they just "train" by climbing more)
# -  +If they take vacations specifically for climbing trips
# -  +Focusing in either sportsclimbing (-or boulder)
# -  +If they climb to compete (-or as a healthy hobby)
# -  +If they focus in the outdoors
# -  +If they train power endurance, explosive power (-or if they train footwork and flex)
# -  +If they wear tighter shoes
# -  +If they project
# -  +If they give the routes several tries
# -  +If they train on a moonboard
# -  +If they climb more routes evey time they climb
# -  +If they warmup (-or if they jump right into the hard stuff)
# 
# Others which have a high correlation but cannot be climbed are...
# -  +How long how they have been climbing / how young they started climbing
# -  +If they have a higher ape index
# 
# Interestingly height, weight and current age did not seem to be very correlated at all, which is a very nice thing. It shows that body type is usually not really a limiting factor for performance.

# ## Feature importance
# 
# We just analyzed correlation, which is good enough in many cases where relations are simply linear, but there are several other ways of finding feature importance where relations might be more complex. To do this, we will try a method called recursive feature elimination, where we start with all the features and remove the features that add noise instead of value (using different models to measure performance). 
# 
# For this experiment lets try linear regression, random forest regression, and gradient boosting regression. It is good to include the 3 to see which behaves better right now, and to combine their results, because models are very dependent on the data size. Right now we only have around 150 registries, but with more data the best predictor might change, and the feature importance would change as well, so it is good to have the models ready to keep it adaptable in case data scales.
# 
# We will test each model using 10-fold cross validation, which means training and testing with 10 different combinations of train/test divisions of the data, to ensure that the model is actually good and not that it simply got a very convenient split of the data.

# In[10]:


X = data.copy().drop(removeVars,axis=1).drop("S_AVG",axis=1)
y = data.copy()["S_AVG"]

#no of features          
tests = [.2, .25, .3, .35]

high_score=0
score_list =[]
topTLR = 0
topFeatsLR = 0
topFeatsPosLR = 0
topFeatsRankLR = 0
featValsLR = 0
for t in tests: 
    print(t)
    #Variable to store the optimum features
    for n in range(1,len(X.columns)):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = t, random_state = 0)
        model = LinearRegression()
        rfe = RFECV(model,n,cv=10)
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score and rfe.n_features_>9 and rfe.n_features_<40):
            topTLR = t
            high_score = score
            nof = rfe.n_features_
            topFeatsPosLR = rfe.support_
            topFeatsLR = X.columns[topFeatsPosLR]
            topFeatsRankLR = rfe.ranking_
            featValsLR = model.coef_
            print("Optimum number of features: %d" %nof)
            print("Score with %d features: %f" % (nof, high_score))
            print(topFeatsLR)
print(topFeatsRankLR)
print(featValsLR)
print(topFeatsLR)


# In[11]:


X = data.copy().drop(removeVars,axis=1).drop("S_AVG",axis=1)
y = data.copy()["S_AVG"]

#no of features          
tests = [.2, .25, .3, .35]

high_score=0
score_list =[]
topTRF = 0
topFeatsRF = 0
topFeatsPosRF = 0
topFeatsRankRF = 0
featValsRF = 0
for t in tests: 
    print(t)
    #Variable to store the optimum features
    for n in range(1,len(X.columns)):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = t, random_state = 0)
        model = RandomForestRegressor(n_estimators=100)
        rfe = RFECV(model,n,cv=10)
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score and rfe.n_features_>9 and rfe.n_features_<40):
            topTRF = t
            high_score = score
            nof = rfe.n_features_
            topFeatsPosRF = rfe.support_
            topFeatsRF = X.columns[topFeatsPosRF]
            topFeatsRankRF = rfe.ranking_
            featValsRF = model.feature_importances_
            print("Optimum number of features: %d" %nof)
            print("Score with %d features: %f" % (nof, high_score))
            print(topFeatsRF)
print(topFeatsRankRF)
print(featValsRF)
print(topFeatsRF)


# In[12]:


X = data.copy().drop(removeVars,axis=1).drop("S_AVG",axis=1)
y = data.copy()["S_AVG"]

#no of features          
tests = [.2, .25, .3, .35]

high_score=0
score_list =[]
topTGB = 0
topFeatsGB = 0
topFeatsPosGB = 0
topFeatsRankGB = 0
featValsGB = 0
for t in tests: 
    print(t)
    #Variable to store the optimum features
    for n in range(1,len(X.columns)):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = t, random_state = 0)
        model = GradientBoostingRegressor(n_estimators=100)
        rfe = RFECV(model,n,cv=10)
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score and rfe.n_features_>9 and rfe.n_features_<40):
            topTGB = t
            high_score = score
            nof = rfe.n_features_
            topFeatsPosGB = rfe.support_
            topFeatsGB = X.columns[topFeatsPosGB]
            topFeatsRankGB = rfe.ranking_
            featValsGB = model.feature_importances_
            print("Optimum number of features: %d" %nof)
            print("Score with %d features: %f" % (nof, high_score))
            print(topFeatsGB)
print(topFeatsRankGB)
print(featValsGB)
print(topFeatsGB)


# Ok now that we have 4 different sources of "feature importance":
# -  Correlation Matrix
# -  Recursive Feature Elimination
# -  -  Linear Regression Feature Optimization
# -  -  Random Forest Feature Optimization
# -  -  Gradient Boosting Feature Optimization
# 
# We can merge them all into a single subset, to keep what we consider the most important variables for this particular problem.

# In[13]:


topFeatsCorr = list(goal2.index)
topFeatsRFE = list(set(list(topFeatsLR)+list(topFeatsRF)+list(topFeatsGB)))
print(len(topFeatsLR))
print(len(topFeatsRF))
print(len(topFeatsGB))
print(len(topFeatsCorr))
print(len(topFeatsRFE))


# So we have about 40 features resulting from the correlation approach, and about 40 features resulting from the RFE approach, we can use these 2 feature sets to create 2 levels of feature importance:
# -  Important features: features that appear on either of the 2 feature sets
# -  Best features: features that appear on both subsets
# 
# *I say about 40 features because this is a stochastic process, you could get different subsets each time you run the code, but they would be very similar.
# 
# It can be useful to keep both because a smaller subset could be better at predicting the outcome, but worse at being understandable, interpretable or actionable by us humans, while a larger subset can allow humans to create strategies based on the variables available.

# In[14]:


importantFeats = list(set(topFeatsRFE+topFeatsCorr))
print(len(importantFeats))


# In[15]:


bestFeats = set(topFeatsRFE).intersection(topFeatsCorr)
print(len(bestFeats))


# In[16]:


topFeatures = pd.DataFrame()
for c in importantFeats:
    topFeatures.loc[0,c]=c
    topFeatures.loc[1,c]=1
    topFeatures.loc[2,c]=0
    
for c in bestFeats:
    topFeatures.loc[2,c]=1
    
topFeatures = topFeatures.T
topFeatures.columns = ["FEATURE","IMPORTANT","BEST"]
topFeatures = topFeatures.reset_index(drop=True)
topFeatures


# In[17]:


topFeatures.to_csv("features.csv",index=False)


# Now we have 2 lists with the variables that have the highest impact on sports climbing performance, these features can be used either by a data scientist to proceed with further data analysis, or by climbers or coaches to see what to focus on in order to improve.
