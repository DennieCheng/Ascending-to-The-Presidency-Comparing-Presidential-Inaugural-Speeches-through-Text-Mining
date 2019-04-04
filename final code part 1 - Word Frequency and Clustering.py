#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:10:50 2019

@author: Dennie Cheng
"""
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
 
#### 1. Word Frequency Distribution

##read all speeches

file_list = os.listdir(os.getcwd())
df = pd.DataFrame(index = list(range(len(file_list))), columns = ("speechList","wordList"))


##clean txt
#Function: remove stopwords
myStopwords = ["can","say","one","way","use","also","howev","tell","will","much","need","take","tend","even","like","particular","rather","said","get","well","make","ask","come","end","first","two","help","often","may","might","see","someth","thing","point","post","look","right","now","think","'ve","'re","anoth","put","set","new","good","want","sure","kind","larg","yes","day","quit","sinc","attempt","lack","seen","awar","littl","ever","moreov","though","found","abl","enough","far","earli","away","achiev","last","never","brief","bit","entir","lot","must","shall"]
txt_stopwords = nltk.corpus.stopwords.words('english') 
myStopwords = myStopwords + txt_stopwords

#word frequency
totalword = []
allspeech = [None]*len(file_list)
newpath = os.path.abspath(os.path.dirname(os.getcwd()))
stemmer = SnowballStemmer("english")
def content_fraction(text):
        content = [w for w in text if w not in myStopwords]
        return content

for i in range(len(file_list)):
    f1 = open(file_list[i])
    txt_raw = f1.read()
    f1.close()
    txt_raw = txt_raw.replace(r"(\|\\n)","")
    txt_raw = txt_raw.lower()                           #change all content to lower case (type:string)
    allspeech[i] = txt_raw
    txt_clean = re.findall(r'\b[a-zA-Z]+\b', txt_raw)   #just select out words

    txt = nltk.text.Text(txt_clean)                     #change to type 'text'
    txt_content = content_fraction(txt)                 #remove stopwords
    lemmatizer = WordNetLemmatizer()
    wordlist = []
    for word in txt_content:
         wordlist.append(stemmer.stem(word))     
    totalword.extend(wordlist)                          #total words
    #write into table
    tit = file_list[i]
    tit = tit[:tit.find('.')]
    df.iloc[i, 0] = tit
    df.iloc[i, 1] = wordlist
    #frequency graph
    fdist = nltk.FreqDist(wordlist)  
    fdist.plot(15,title = tit)   

    
   
    
         
    
#### 2. Clustering Presents' speeches


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
    

#build tf-idf matrix
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words=txt_stopwords,
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))


tfidf_matrix = tfidf_vectorizer.fit_transform(allspeech) #fit the vectorizer to all speeches

# (10, 1008) means the matrix has 10 rows and 1008 columns
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
len(terms)

num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

#combine cluster index into the table of speeches
df_cluster = df
df_cluster['clusterNum'] = clusters
df_cluster = df_cluster.sort_values("clusterNum")

#top words in each cluster
#topWord0 = df_cluster.iloc[0, 1] + df_cluster.iloc[1, 1] + df_cluster.iloc[2, 1] + df_cluster.iloc[3, 1]
#topWord0g = nltk.FreqDist(topWord0)
#topWord0g.plot(10,title = "cluster0") 


#calculate cosine similarity
similarity_distance = 1 - cosine_similarity(tfidf_matrix)
print(type(similarity_distance))
print(similarity_distance.shape)



# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify "random_state" so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(similarity_distance)  # shape (n_components, n_samples)

print(pos.shape)
print(pos)

xs, ys = pos[:, 0], pos[:, 1]

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#66a61e', 4: '#e7298a'}

#set up cluster names using a dict
cluster_names = {0: '1', 
                 1: '2', 
                 2: '3', 
                 3: '4',
                 4: '5'
                 }

dfc = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=df.iloc[0:,0])) 
dfc.to_csv('clusterMatrix.csv')

groups = dfc.groupby('label')

fig, ax = plt.subplots(figsize=(13, 8))


for name, group in groups:
 
    print("group name " + str(name))
    print(group)
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point




for i in range(len(dfc)):
    ax.text(dfc.iloc[i]['x'], dfc.iloc[i]['y'], dfc.iloc[i]['title'], size=10)  
    

ax.show()

