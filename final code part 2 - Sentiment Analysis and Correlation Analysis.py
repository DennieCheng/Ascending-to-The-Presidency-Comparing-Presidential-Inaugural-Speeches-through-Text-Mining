#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 18:39:34 2019

@author: NINI
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:35:29 2019

@author: NINI
"""
import os
import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt


#### 3. Sentiments analysis

###3.1 determine sentiments score
#define stopwords
myStopwords = ["can","say","one","way","use","also","howev","tell","will","much","need","take","tend","even","like","particular","rather","said","get","well","make","ask","come","end","first","two","help","often","may","might","see","someth","thing","point","post","look","right","now","think","'ve","'re","anoth","put","set","new","good","want","sure","kind","larg","yes","day","quit","sinc","attempt","lack","seen","awar","littl","ever","moreov","though","found","abl","enough","far","earli","away","achiev","last","never","brief","bit","entir","lot","must","shall"]
txt_stopwords = nltk.corpus.stopwords.words('english') 
myStopwords = myStopwords + txt_stopwords


#define a function to generate the plority value of every sentense of Reagan's and Trump's speech
def sentsScore(President, Sents):
    dfSents = pd.DataFrame(index = range(len(Sents)), columns = ("President", "WordList", "score"))
    for j in range(len(Sents)):
        dfSents.iloc[j, 0] = President
        dfSents.iloc[j, 1] = [ w.lemmatize() for w in Sents[j].words if w not in myStopwords]
        dfSents.iloc[j, 2] = Sents[j].sentiment.polarity
    dfSents = dfSents.drop(dfSents.index[len(Sents)-1])
    dfSents = dfSents.drop(dfSents.index[0])
    return dfSents
 



#Read all speeches list
file_list1 = os.listdir(os.getcwd())

#sentiments score of each sentence
allScore = pd.DataFrame()


for i in range(len(file_list1)):
    f1 = open(file_list1[i])
    txt_raw = f1.read()
    f1.close()
    President = file_list1[i][:file_list1[i].find('.')]
    txt_raw = txt_raw.replace(r'(\\|\\\n)'," ")
    txt_raw = txt_raw.lower()  
    blob = TextBlob(txt_raw)
    Sents = blob.sentences 
    allScore = pd.concat([allScore, sentsScore(President, Sents)])


allScore['Sentiment'] = [None]*allScore.shape[0]
for i in range(allScore.shape[0]):
    if allScore.iloc[i, 2] == 0:
        allScore.iloc[i, 3] = "Neutral"
    elif allScore.iloc[i, 2] < 0:
        allScore.iloc[i, 3] = "Negative"
    else:
        allScore.iloc[i, 3] = "Positive"

#write into csv for further analysis in Tableau
allScore.to_csv('AllPresident sentsScore.csv')




###3.2 Visualize sentiments
#define a function for drawing positive/negative words
def wordcloud_draw(data, name = 'fig', color = 'black'):
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(data)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('%s.jpg' % name)
    plt.close()
    

##Trump
dfTrump2017 = allScore[allScore['President'] == '2017 DonaldTrump']    

TrumpPositive = []
TrumpNegative = []
for i in range(dfTrump2017.shape[0]):
    if dfTrump2017['Sentiment'].iloc[i] == 'Positive':
        TrumpPositive.extend(dfTrump2017['WordList'].iloc[i])
    elif dfTrump2017['Sentiment'].iloc[i] == 'Negative':
        TrumpNegative.extend(dfTrump2017['WordList'].iloc[i])
    else:
        continue

    
TrumpPositive = str(TrumpPositive).replace("'", " ")    
TrumpNegative = str(TrumpNegative).replace("'", " ")   
wordcloud_draw(TrumpPositive, 'TrumpPositive', 'white')
wordcloud_draw(TrumpNegative, 'TrumpNegative')

##Reagan
dfReagan1981 = allScore[allScore['President'] == '1981 RonaldReagan']  
dfReagan1985 = allScore[allScore['President'] == '1985 RonaldReagan']  

#1981
ReaganPositive1981 = []
ReaganNegative1981 = []
ReaganNeutral1981 = []
for i in range(dfReagan1981.shape[0]):
    if dfReagan1981['Sentiment'].iloc[i] == 'Positive':
        ReaganPositive1981.extend(dfReagan1981['WordList'].iloc[i])
    elif dfReagan1981['Sentiment'].iloc[i] == 'Negative':
        ReaganNegative1981.extend(dfReagan1981['WordList'].iloc[i])
    else:
        ReaganNeutral1981.extend(dfReagan1981['WordList'].iloc[i])

ReaganPositive1981 = str(ReaganPositive1981).replace("'", " ")    
ReaganNegative1981 = str(ReaganNegative1981).replace("'", " ")   
ReaganNeutral1981 = str(ReaganNeutral1981).replace("'", " ") 
wordcloud_draw(ReaganPositive1981, '1981ReaganPositive', 'white')
wordcloud_draw(ReaganNegative1981, '1981ReaganNegative')
wordcloud_draw(ReaganNeutral1981, 'brown')

#1985
ReaganPositive1985 = []
ReaganNegative1985 = []
ReaganNeutral1985 = []
for i in range(dfReagan1985.shape[0]):
    if dfReagan1985['Sentiment'].iloc[i] == 'Positive':
        ReaganPositive1985.extend(dfReagan1985['WordList'].iloc[i])
    elif dfReagan1985['Sentiment'].iloc[i] == 'Negative':
        ReaganNegative1985.extend(dfReagan1985['WordList'].iloc[i])
    else:
        ReaganNeutral1985.extend(dfReagan1985['WordList'].iloc[i])

ReaganPositive1985 = str(ReaganPositive1985).replace("'", " ")    
ReaganNegative1985 = str(ReaganNegative1985).replace("'", " ")   
ReaganNeutral1985 = str(ReaganNeutral1985).replace("'", " ") 
wordcloud_draw(ReaganPositive1985, '1985ReaganPositive', 'white')
wordcloud_draw(ReaganNegative1985, '1985ReaganNegative')
wordcloud_draw(ReaganNeutral1985, 'brown')


##1989 George Bush
dfBush1989 = allScore[allScore['President'] == '1989 GeorgeBush']    

BushPositive = []
BushNegative = []
for i in range(dfBush1989.shape[0]):
    if dfBush1989['Sentiment'].iloc[i] == 'Positive':
        BushPositive.extend(dfBush1989['WordList'].iloc[i])
    elif dfBush1989['Sentiment'].iloc[i] == 'Negative':
        BushNegative.extend(dfBush1989['WordList'].iloc[i])
    else:
        continue

    
BushPositive = str(BushPositive).replace("'", " ")    
BushNegative = str(BushNegative).replace("'", " ")   
wordcloud_draw(BushPositive, '1989BushPositive', 'white')
wordcloud_draw(BushNegative, '1989BushNegative')

#### 4. correlation of sentiment score and economic index
corDf = pd.read_excel('association analysis.xlsx')
corDf = corDf[['Negative Avg. Score', 'Positive Avg. Score', 'Avg.GDP', 'Avg.UnemploymentRate']]
corr = corDf.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(corDf.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(corDf.columns)
ax.set_yticklabels(corDf.columns)
plt.show()
