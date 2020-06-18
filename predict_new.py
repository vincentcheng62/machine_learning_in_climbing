#!/usr/bin/env python
# coding: utf-8

# # Inference, forecast and optimization
# 
# Ok, the final step in this exercise will be to use the data and variables to start infering, forecasting and optimizing the result of a target variable, instead of just trying to describe what already is happening.
# 
# To do this, we can create a model based on all the data available (common approach, but not the best one if you already know that different subjects behave differently) or we can create models for a specific subset of the entire population.
# 
# This notebook will showcase both.
# 
# *One thing to note is that I do not expect the models to have a very good performance, it is mainly illustrative, because we are dealing with a very small database. Having only ~150 climbers, to cover the entire range of grades from 5.8 to 5.15d, gives very few examples of each grade for the algorithms to properly learn patterns. But replicating this same strategy with a database that is tens, hundreds or thousands of times bigger will yield much better results.

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.neighbors import NearestNeighbors, KDTree


# In[3]:


fullData = pd.read_csv("filteredData_new.csv").replace(np.nan,0)
normData = pd.read_csv("normData_new.csv").replace(np.nan,0)

mixedData = normData.copy()
mixedData.iloc[:,-17:] = fullData.iloc[:,-17:]

features = pd.read_csv("features.csv")
testImportant = mixedData.loc[:,features.loc[features.loc[:,"IMPORTANT"]==1,"FEATURE"]]
testImportant["S_AVG"]=fullData["S_AVG"]
testBest = mixedData.loc[:,features.loc[features.loc[:,"BEST"]==1,"FEATURE"]]
testBest["S_AVG"]=fullData["S_AVG"]

print(testBest)


# In[4]:


testImportant.describe()


# In[5]:


testBest.describe()


# Lets try the general approach with both the "best" and the "important" datasets to see which performs best, then we can go into the subset modeling only with the better performing one.
# 
# Lets define the trainModel function since it will be used several times, then lets use it.

# In[6]:


def trainModel(model,db,indexes,tests,goal):
    X = db.copy().drop(goal,axis=1)
    X = X.loc[indexes,:]
    y = db.copy()[goal]
    y = y[indexes]

    high_score=0
    score_list =[]
    topTRF = 0
    topFeatsRF = 0
    topFeatsPosRF = 0
    topFeatsRankRF = 0
    featValsRF = 0
    topModel = RandomForestRegressor(n_estimators=100)
    topX_train = 0
    topX_test = 0
    topy_train = 0
    topy_test = 0

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
                topModel.fit(X_train_rfe,y_train)
                topX_train = X_train
                topX_test = X_test
                topy_train = y_train
                topy_test = y_test
                print("Score with %d features: %f" % (nof, high_score))
                
    return topModel, topFeatsRF, topFeatsPosRF, topX_train, topX_test, topy_train, topy_test




# In the examples above, the models use something between 65% and 75% of the data in order to train the model, and something between 25% to 35% of the data to evaluate how it performs. As specified at the beggining, having so little data, great performance was not expected. Right now using the "best" features we got models that can predict a climber's performance with an average error of around +-3 grades (that is, predicting between 5.8 and 11a for 5.10b climber), and errors as large as +-7 grades max, which while not great at all, it is actually better than I was hoping for in this example having only 150 climbers to learn from.
# 
# As foreshadowed (even if it is initially counter-intuitive) the model using only the best features performs better than using all the important features. One could initially find it weird that using less information (around 25 vs 60 variables) would yield better results, but many times data is noisy with misleading variables that can distract from the patterns-carrying variables.
# 
# Instead of showcasing how to use these models to show how a specific climber should/could be performing, or to show how to propose strategies or plans of action to achieve golas one might have, I will first show how one could prefer to group first and perform modeling on smaller groups.
# 
# ## Clustering and cluster analysis
# 
# The predictive model we already created uses information gathered from all the climbers in order to predict how a climber would perform given his profile, his training habits, his mentality of climbing, etc. However, the model as we have it right now does not discriminate, so it will use information from a 1.95 m, 80 kg 48 year old male climber as an example of what is expected from a 1.55 m, 42 kg, 19 year old girl, and that miiiiight not be the best approach.
# 
# Data Scientists usually deal with this problem by performing clustering or grouping. There are many different approaches to do this. When certain domain knowledge is possessed one could divide them manually (for example divide male and female, or divide 5.10, 5.11, 5.12, 5.13, 5.14, or divide 40-50kg, 50-60kg, etc.), when there is no clear way of how the data should be divided, one could rely on machine learning algorithms that perform automatic clustering. In case you want to learn one, the most easy to learn and implement would probably be K-means. In K-means you tell the algorithm how many groups you want, and it will find the best way it could be divided into that many groups.
# 
# I will do a combination of these approaches. Since the data size is very small, and I possess some domain knowledge, I could find registry-specific-clustering. That is, instead of finding general groups of datapoints that are similar to each other, finding the datapoints that are the most similar to a particular datapoint. As an analogy to this climbing example, instead of dividing the data into groups and then finding the group that would include John, I will create a group with the climbers that are specifically the most similar to John. In data science we call this K-NN (the K - nearest neighbors) and there are algorithms in python that let us do it easily.
# 
# "Nearest" is defined based on difference between their variables, so we first need to define the variable-subset profile we would like the algorithm to compare, instead of letting it find the nearest neighbors using all the variables.
# 
# Simply for example case, I will propose the following variables:

# In[10]:


knnProfileVars = ["AGE","YRS_CLIMBING","HEIGHT","APEINDEX","WEIGHT","BMI","B_AVG","S_AVG"]
knnData = normData[knnProfileVars]
knnData.describe()


# The reason I propose this variables is because it contains their current level, and descriptive variables that they cannot change, they simply describe their current phisique and experience. In order to create groups of climbers with similar phisiques, experience and current performance, regardless of the way they train, how often they climb, how they approach improvement, what they eat, etc. because those are the variables that the climber can actually change in order to produce a change in their performance.
# 
# Now lets use a KDTree (the algorithm inside the KNN algorithm) to find the nearest neighbors of a random climber, lets say the 10th climber in the list.

# In[22]:


climberID = 155
randomClimber = knnData.loc[climberID,:]
tree = KDTree(knnData)     
dist, ids = tree.query([randomClimber], k=int(len(knnData.index)/3))

closestClimbers = knnData.loc[ids[0],:]

comparison = pd.DataFrame()
comparison["SUBJECT_CLIMBER"]=randomClimber
comparison["AVERAGE_CLIMBER"]=knnData.mean()
comparison["KNN_CLOSEST_AVG"]=closestClimbers.mean()

print("After finding the",str(int(len(knnData.index)/3)),"nearest neighbors we see this behavior in the data distribution")
#display(comparison)


# As you can see, the new group is conformed of climbers with a profile that is closer in similarity to the climber we care about. So now, learning the effect of particular actions on perforance makes more sense, since climbers with very similar bodies and experience intuitively would benefit from similar actions. So if a climber with a similar profile had certain benefit from an action, it would suggest that you probably should too.
# 
# So lets get back to a predictive modeling using only these subsets.

# In[30]:


neighbors = ids[0][1:]
dataG = testBest.copy().drop(climberID,axis=0)
modelG = RandomForestRegressor(n_estimators=100)
#tests = [.05, .075, .1, .15]
tests = [.05]

modelG,featsG,featspG,x_trG,x_tsG,y_trG,y_tsG = trainModel(modelG,dataG,neighbors,tests,"S_AVG") 


# Ok so now we have a model trained specifically for the climber of interest. In this case it probably has similar or lower performance than the previous model we trained using all the climbers. This is mainly because we are trying to have it learn complex relationships between dozens of variables using only 50 examples. Even if this data grouping is of better quality than using the entire dataset, the number of registries is still the major limitation at hand. If more data were available, the models' performances would improve dramatically.
# 
# ## Optimization
# 
# So what can we do with this model? How can we use it to guide improvement?
# Well, there are various ways. Since we now know the impact of each variable on the target variable, we can try to tweek the variables to optimize the outcome.

# In[44]:


#predictsG = modelG.predict(x_tsG[featsG])
#compareG = y_tsG.copy().reset_index(drop=True)

predictsG = modelG.predict([testBest.copy().drop("S_AVG",axis=1).loc[climberID,featsG]])
compareG = testBest.copy().loc[climberID,'S_AVG']

print("ID\tPred\tReal\tError")
for i,p in enumerate(predictsG):
    print(str(i)+"\t"+str(p)+"\t"+str(compareG)+"\t"+str(compareG-p))


# Right now, our climber10 has a real climbing average grade of 3.5 (between 5.10b/6a+ and 5.10c/6b) and our algoritm predicts it rather acurrately (it predicts 3.9 which is still between 5.10b/6a+ and 5.10c/6b) using only the variables show below (except "S_AVG"). 

# In[60]:


targetClimber = testBest.copy().loc[climberID,featsG]
targetClimber["S_AVG"]=testBest.copy().loc[climberID,"S_AVG"]
targetClimberView = fullData.copy().loc[climberID,featsG]
targetClimberView["S_AVG"] = fullData.copy().loc[climberID,"S_AVG"]
viewClimber = pd.DataFrame()
viewClimber["NORM"]=targetClimber
viewClimber["VIEW"]=targetClimberView
viewClimber


# This by itself is already valuable since it allows for a correct assessing of what a climber's performance should/could be based on their profile. However, its also gives us the possibility of exploring what the performance would be if one of those variables were to change.
# 
# Just stating some characteristics out of the lot, right now our climber...
# -  Climbs 3 days per week
# -  Climbs 3 routes per climbing day
# -  Has been climbing for 3 years
# -  Trains by simply climbing more instead of creating a training plan by himself/herself
# -  Attempts hard routes twice a day
# 
# Based on that lets explore what his/her performance would be with small changes, using the model...
# -  A: Climb 5 days a week instead of 3
# -  B: CLimb 6 routes a day instead of 3
# -  C: Climb for 6 more months
# -  D: Switch to following a personal training plan instead of just climbing
# -  E: Attempts hard routes 3 times a day
# -  F: Combine A, B and E (increase amount of climbing and attempts)
# -  G: Combine C, D (start a personal training plan and follow it for 6 months) 

# In[80]:


#predictsG = modelG.predict(x_tsG[featsG])
#compareG = y_tsG.copy().reset_index(drop=True)

print("Real Now:\t"+str(compareG))
print("Pred Now\t"+str(predictsG[0]))

tester = targetClimber.copy()
tester["CLIMBDAYS"] = .875 #Corresponds to 5 days after normalization
predictsTest = modelG.predict([tester.drop("S_AVG")])
print("Test A\t\t"+str(predictsTest[0])+"\tImprove:\t"+(str(predictsTest[0]-compareG)))

tester = targetClimber.copy()
tester["ROUTEQTY"] = .2 #Corresponds to 6 routes after normalization
predictsTest = modelG.predict([tester.drop("S_AVG")])
print("Test B\t\t"+str(predictsTest[0])+"\tImprove:\t"+(str(predictsTest[0]-compareG)))

tester = targetClimber.copy()
tester["YRS_CLIMBING"] = .09836 #Corresponds to 3.5 years after normalization
predictsTest = modelG.predict([tester.drop("S_AVG")])
print("Test C\t\t"+str(predictsTest[0])+"\tImprove:\t"+(str(predictsTest[0]-compareG)))

tester = targetClimber.copy()
tester["TRAIN_CLIMB"] = 0 
tester["TRAIN_SELF"] = 1 
predictsTest = modelG.predict([tester.drop("S_AVG")])
print("Test D\t\t"+str(predictsTest[0])+"\tImprove:\t"+(str(predictsTest[0]-compareG)))

tester = targetClimber.copy()
tester["ATTEMPTS"] = .5 #Corresponds to 3 attempts after normalization
predictsTest = modelG.predict([tester.drop("S_AVG")])
print("Test E\t\t"+str(predictsTest[0])+"\tImprove:\t"+(str(predictsTest[0]-compareG)))

tester = targetClimber.copy()
tester["CLIMBDAYS"] = .875 #Corresponds to 5 days after normalization
tester["ROUTEQTY"] = .2 #Corresponds to 6 routes after normalization
tester["ATTEMPTS"] = .5 #Corresponds to 3 attempts after normalization
predictsTest = modelG.predict([tester.drop("S_AVG")])
print("Test F\t\t"+str(predictsTest[0])+"\tImprove:\t"+(str(predictsTest[0]-compareG)))

tester = targetClimber.copy()
tester["YRS_CLIMBING"] = .09836 #Corresponds to 3.5 years after normalization
tester["TRAIN_CLIMB"] = 0 
tester["TRAIN_SELF"] = 1 
predictsTest = modelG.predict([tester.drop("S_AVG")])
print("Test G\t\t"+str(predictsTest[0])+"\tImprove:\t"+(str(predictsTest[0]-compareG)))


# So with this examples we can see what the expected outcome of each possible strategy would be, based on historic factual data of what has worked for other climbers. This could for example guide the climber or coach in which is the most efficient path towards improvement, as one could not simply do everything at the same time, change should usually be gradual.
# 
# In this example, it seems like if climbing 6 routes instead of 3 routes whenever the climber goes to the crag is what would bring the most improvement, almost all the way to 5.10d/6b+ and it is something well within the climber's control.
# 
# 
# ## Reverse engineering
# 
# In this example, the different proposed strategies were completely arbitrary, but since we already have the information of the other climbers similar to our climber number 10, we can use that information to suggest a potential approach.

# In[85]:


betterClosestClimbers = testBest.copy().loc[closestClimbers.index[1:],featsG]
betterClosestClimbers["S_AVG"]=testBest.copy().loc[closestClimbers.index[1:],"S_AVG"]
betterClosestClimbers


# From this list of closest climbers, we could focus simply on those that have a higher performance than climber number 10, in order to see what they are doing and imitate them. So we simply filter those and compute their mean.

# In[91]:


betterClosestClimbers = betterClosestClimbers[betterClosestClimbers["S_AVG"]>targetClimber["S_AVG"]]
improveClimb = betterClosestClimbers.mean()
hardImprove = betterClosestClimbers.max()
viewImprove = pd.DataFrame()
viewImprove["CURRENT"]=targetClimber
viewImprove["IMPROVE"]=improveClimb
viewImprove["MAXGAIN"]=hardImprove
viewImprove


# Ok, by seeing what the "better" climbers with her same profile do, we could come out with individual strategies and try them out just like we did before. But with this we can see the "sweetspot" value for each variable and the expected increase assuming the climber had those change of habits.
# 
# The last thing we just displayed is indirectly some kind of reverse engineering, basically seeing the output and then figuring out how to replicate it. That is one way of doing it based on what other climbers have done, but we could also use computing power to try to mathematically figure out which the best approach would be.
# 
# Lets try that out to see what the optimal number of routes per day would be for a climber at that current state.

# In[93]:


maxScore = 0
maxClibs = 0
for i in range(32):   
    tester = targetClimber.copy()
    tester["ROUTEQTY"] = i/31
    predictsTest = modelG.predict([tester.drop("S_AVG")])
    if predictsTest[0] > maxScore:
        maxScore = predictsTest[0]
        maxClimbs = i
        print("By climbing",maxClimbs,"climbs a day, climber would climb",maxScore)


# In this example after forecasting what the grade would be by climbing 1-31 routes a day, the algorithm found that the optimal would be reached by climbing 8 routes a day assuming nothing else changes.
# 
# In this case we are only looking at a single variable to optimize, in reality you could try to optimize all variables at the same time because their effects are interconnected, but that becomes more complex or unfeasible to try in a "brufe force" (what we just did of trying 0 to 31), because there are too many potential combinations to try them. There are many other algorithms for this such as genetic algorithms, greedy algorithms, simulated annealing, etc. but we will not explore them at this time as this was meant mainly as an introductory exercise.
# 
# ## Conclusions
# 
# In general, this covers pretty much all the steps of a basic/generic data science workflow. Evey step showcased in these notebooks can be made much more complex in order to go into as much detail as desired, but you can work on that by yourself once you are involved in other projects. Data science is much more about creativity and curiosity than it is of hardcore programming. As you saw in these exercises, most of the interesting insights were found because there was a hunch and we simply explored it with rather simple code to see if we could find anything interesting in there. You don't need to be a programming expert to get into data scientist, I actually do not consider myself a very good programmer, I have pretty basic skills and simply use that knowledge + intuition as a tool to play with data, and ask for help either from google or other colleagues whenever I have an idea I want to implement yet lack the technical prowess to do so.
# 
# Right now some of the processes showcased here are extremely inefficient and would actually not work with larger datasets due to the amount of time or memory they would require, but they are very easy to understand for someone who is just getting into data science. In order to tackle those other scenarios that I mention one must use either a form of parallel computing such as scala/pyspark, or more efficient (yet usually complex) algorithms.
# 
# Numpy, pandas, scikit_learn, google and stackoverflow are your best friends as a data scientist. Whenever you don't know how to use a tool, or you get errors you don't understand simply googling would either outright answer you guide you towards a good course of action.
# 
# I hope this was useful, understandable, interesting and easy to follow. And I hope it motivates you to get more into data science if you were already curious about it but didn't know how to start. There are many resourses online for you to get up to speed if you wish to pursue this path.
# 
# ### Note
# 
# Data science is very dependent on the amount and quality of data. This was performed using only 156 responses (extremely little database) of a survey created by myself, someone with data science knowledge but without any proper knowledge about climbing or sports science, nor the tools to get more detailed quantitative information (rather bad quality of data). With either more responses, or with the help of an actual sports professional to create a more appropiate survey to gather actually useful information, the performance of these kinds of analysis will greatly improve, the results would be more evident and the findings become much more useful.
# 
# This notebook is a static shot of the code from the last time I ran it, if you download the code and run it in your own computer, the results may change and the "hardcoded examples" might stop making any sense because several of the processes are stochastic. This means that they have an intrinsic randomness in their inner workins that causes them to yield different outputs every time you run them, but usually quite similar. This is both good and bad, a good thing is that you can run these processes many times, storing the best result (as we did in the "trainModel" methods of this function) to get very good performances, but the bad thing is that they might not be directly replicable if the random seed was not stored.

# In[ ]:




