# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:24:58 2022

@author: 616819
"""

###CFB bowl picks

import pandas as pd
import numpy as np
import random

#%%
results=pd.read_csv("2022results.csv")
statistics=pd.read_csv("2022stats.csv")


offensiveStats=pd.read_html("https://www.espn.com/college-football/stats/team/_/view/offense")
defensiveStats=pd.read_html("https://www.espn.com/college-football/stats/team/_/view/defense")
specialStats=pd.read_html("https://www.espn.com/college-football/stats/team/_/view/special")

#%%
x=[defensiveStats[0].columns[0]]+list(defensiveStats[0][defensiveStats[0].columns[0]])
defensiveStats[1]['team']=x

defStat=defensiveStats[1]

x=[offensiveStats[0].columns[0]]+list(offensiveStats[0][offensiveStats[0].columns[0]])
offensiveStats[1]['team']=x

offStat=offensiveStats[1]

x=[specialStats[0].columns[0]]+list(specialStats[0][specialStats[0].columns[0]])
specialStats[1]['team']=x

specStat=specialStats[1]

#%%

names=pd.read_csv("NameDict.csv")
CFBtoESPN=dict(zip(names['CFBname'], names['ESPNname']))


teamStats={}

for team in np.unique(results['home_team']):
    print(team)
    stats=statistics[statistics['team']==team]
    try:
        offstats=offStat[offStat['team']==CFBtoESPN[team]]
        defstats=defStat[defStat['team']==CFBtoESPN[team]]
    except KeyError:
        continue
    dfrow={'PointsPerGame': [offstats[('Points','PTS/G')].values[0]],
           'PassingYardsPerGame' : [offstats[('Passing','YDS/G')].values[0]],
           'RushingYardsPerGame' : [offstats[('Rushing','YDS/G')].values[0]],
           'PassingYardsPerAttempt' : [(stats.loc[stats['statName']=='netPassingYards', 'statValue'].values[0])/(stats.loc[stats['statName']=='passAttempts', 'statValue'].values[0])],
           'RushingYardsPerAttempt' : [(stats.loc[stats['statName']=='rushingYards', 'statValue'].values[0])/(stats.loc[stats['statName']=='rushingAttempts', 'statValue'].values[0])],
           'RushingYardsAllowedPerGame' : [defstats[('Rushing','YDS/G')].values[0]],
           'PassingYardsAllowedPerGame' : [defstats[('Passing','YDS/G')].values[0]],
           'PointsAllowedPerGame' : [defstats[('Points','PTS/G')].values[0]],
           'FirstDowns' : [stats.loc[stats['statName']=='firstDowns', 'statValue'].values[0]],
           'ThirdDownSuccess' : [(stats.loc[stats['statName']=='thirdDownConversions','statValue'].values[0])/(stats.loc[stats['statName']=='thirdDowns','statValue'].values[0])],
           'FourthDownSuccess' : [(stats.loc[stats['statName']=='fourthDownConversions','statValue'].values[0])/(stats.loc[stats['statName']=='fourthDowns','statValue'].values[0])],
           'FumblesPerGame' : [(stats.loc[stats['statName']=='fumblesLost','statValue'].values[0])/(stats.loc[stats['statName']=='games','statValue'].values[0])],
           'FumblesRecoveredPerGame' : [(stats.loc[stats['statName']=='fumblesRecovered','statValue']).values[0]/(stats.loc[stats['statName']=='games','statValue'].values[0])],
           'InterceptionsPerGame' : [(stats.loc[stats['statName']=='passesIntercepted','statValue'].values[0])/(stats.loc[stats['statName']=='games','statValue'].values[0])],
           'InterceptionsRecoveredPerGame' : [(stats.loc[stats['statName']=='interceptions','statValue'].values[0])/(stats.loc[stats['statName']=='games','statValue'].values[0])],
           'YardsPerReturn' : [((stats.loc[stats['statName']=='kickReturnYards','statValue'].values[0]) +(stats.loc[stats['statName']=='puntReturnYards','statValue'].values[0]))/((stats.loc[stats['statName']=='kickReturns','statValue'].values[0])+(stats.loc[stats['statName']=='puntReturns','statValue'].values[0]))],
           'PenaltyYardsPerGame' : [(stats.loc[stats['statName']=='penaltyYards','statValue'].values[0])/(stats.loc[stats['statName']=='games','statValue'].values[0])],
           'SacksPerGame' : [(stats.loc[stats['statName']=='sacks','statValue'].values[0])/(stats.loc[stats['statName']=='games','statValue'].values[0])]}
    teamStats[team]=dfrow
    
#%%
first=True
for index, game in results.iterrows():
   homea=random.randint(0, 1)
   if homea:
       ateam=game["home_team"]
       bteam=game["away_team"]
       apoints=game["home_points"]
       bpoints=game["away_points"]
       ahome=1-(game['neutral_site'])
       bhome=0
   else:
       ateam=game["away_team"]
       bteam=game["home_team"]
       apoints=game["away_points"]
       bpoints=game["home_points"]
       ahome=0
       bhome=1-(game['neutral_site'])
       
   if np.isnan(apoints) or np.isnan(bpoints):
       continue
   try:
       oldastats=teamStats[ateam]
       oldbstats=teamStats[bteam]
   except KeyError:
       continue
   astats={"a" + key : oldastats[key] for key in oldastats}
   bstats={"b" + key : oldbstats[key] for key in oldbstats}
   otherstats={"aHome" : [ahome],
               "bHome" : [bhome],
               "aPoints" : [apoints],
               "bPoints" : [bpoints]}
   gameDict={**astats, **bstats, **otherstats}
   if first:
       gameDf=pd.DataFrame(data=gameDict)
       first=False
   else:
       gameDf=gameDf.append(pd.DataFrame(data=gameDict))
       
gameDf=gameDf.reset_index()
del gameDf['index']
#%%

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

#%%

train_dataset = gameDf.sample(frac=0.8, random_state=0)
test_dataset = gameDf.drop(train_dataset.index)

x_train=train_dataset.loc[:, [x not in ["aPoints", "bPoints"] for x in train_dataset.columns]]
y_train=train_dataset.loc[:, [x in ["aPoints", "bPoints"] for x in train_dataset.columns]]
x_test=test_dataset.loc[:, [x not in ["aPoints", "bPoints"] for x in test_dataset.columns]]
y_test=test_dataset.loc[:, [x in ["aPoints", "bPoints"] for x in test_dataset.columns]]

normalizer = layers.Normalization(input_shape=[38,1], axis=None)
normalizer.adapt(x_train)

model=tf.keras.Sequential([
    layers.Dense(128, input_dim=38, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(2, input_dim=38)])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

print(model.summary())


model.fit(x_train,y_train, validation_split=0.2, batch_size=64, verbose=True, epochs=100)

#%%

yhat=model.predict(x_test)
resid=yhat-y_test
squaredError=np.square(resid).to_numpy().sum()
meanSquaredError=squaredError/(resid.shape[0]*resid.shape[1])

#%%

def predictGame(homeTeam, awayTeam, neutralSite=True, notes = ''):
    try:
        oldhomestats=teamStats[homeTeam]
        oldawaystats=teamStats[awayTeam]
    except KeyError as E:
        print("Team not in list")
        raise E
    astats={"a" + key : oldhomestats[key] for key in oldhomestats}
    bstats={"b" + key : oldawaystats[key] for key in oldawaystats}
    otherstats={"aHome" : [1-neutralSite],
                "bHome" : [0]}
    gameDict={**astats, **bstats, **otherstats}
    gameDf=pd.DataFrame(data=gameDict)
    yhat1=model.predict(gameDf, verbose=False)
    
    bstats={"b" + key : oldhomestats[key] for key in oldhomestats}
    astats={"a" + key : oldawaystats[key] for key in oldawaystats}
    otherstats={"aHome" : [0],
                "bHome" : [1-neutralSite]
                }
    gameDict={**astats, **bstats, **otherstats}
    gameDf=pd.DataFrame(data=gameDict)
    yhat2=model.predict(gameDf, verbose=False)
    
    homeTeamPred=(yhat1[0][0]+yhat2[0][1])/2
    awayTeamPred=(yhat1[0][1]+yhat2[0][0])/2
    
    resultString= notes + " -- " + homeTeam + ": " + f'{homeTeamPred:3.2f}' +" -- "+ awayTeam + ": " + f'{awayTeamPred:3.2f}'
    print(resultString)
    return((homeTeamPred, awayTeamPred))

predictGame("Michigan", "Georgia", neutralSite=True)

#%%

bowlSched=pd.read_csv("2022bowlSched.csv")
for index, game in bowlSched.iterrows():
    try:
        predictGame(game['home_team'], game['away_team'], neutralSite=True, notes=game['notes'])
    except KeyError:
        print("Cannot Compute")
        continue
    
    
