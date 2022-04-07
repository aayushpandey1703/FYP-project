import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy

import warnings

warnings.filterwarnings("ignore")
nlp=spacy.load("en_core_web_sm")
import pickle
import tensorflow
from tensorflow import keras
file_to_read = open("/content/drive/MyDrive/FYP/stored_object.pickle", "rb")
z=pickle.load(file_to_read)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import geopandas as gpd


st.set_page_config(layout="wide",menu_items=None)

t=Tokenizer(num_words=50000,lower=True)
t.fit_on_texts(z)
x=t.texts_to_sequences(z)
x=sequence.pad_sequences(x,maxlen=250)

model=keras.models.load_model("/content/drive/MyDrive/FYP/my_model.h5")

def tweet_scrape(num_tweets):         # scraped tweets
  import requests 
  import json
  import tweepy
  bearer_token='AAAAAAAAAAAAAAAAAAAAAJbiNgEAAAAA9XfDS3YRCsEWv9GoitFHFTCuu9I%3DFdYzAbd2GvMgStnVTDVlnD1TVMARk4w306nGUt9YHNoT2AkUPQ'
  client=tweepy.Client(bearer_token=bearer_token)
  tweetDf={'data':[],'users':[]}

  def getUser(ids):
    users=client.get_users(ids=ids,user_fields=['name','location'])
    tweetDf['users']+=users.data
      

  def search_twitter(query, tweet_fields ,bearer_token):
    ids=[]
    tweets=client.search_recent_tweets(query=query,tweet_fields=tweet_fields,expansions=['author_id'],max_results=100) 
    tweetDf['data']=tweets.data
    since_id=tweetDf['data'][-1].id

    for i in range(num_tweets):
      tweets=client.search_recent_tweets(query=query,tweet_fields=tweet_fields,expansions=['author_id'],since_id=since_id,max_results=100)      
      tweetDf['data']+=tweets.data
      since_id=tweetDf['data'][-1].id

    for i in tweetDf['data']:
      ids.append(i['author_id'])
    for i in range(2):
      try:
        getUser(ids[100*i:100*i+100])
      except:
        getUser(ids[500:len(ids)])
    return tweetDf

  query="i want to buy iphone -is:retweet OR i hate iphone -is:retweet OR i love iphone -is:retweet"
  tweet_fields=['text','author_id','created_at','lang']
  json_response=search_twitter(query=query, tweet_fields=tweet_fields,bearer_token=bearer_token)
  return json_response

def tweet_df(json_response):
  data={'username':[],'name':[],'tweet':[],'location':[]}
  for i in range(len(json_response['data'])):
    try:
      doc=nlp(json_response['users'][i]['location'])
      k=0
      for token in doc:
        if token.ent_type_=="GPE":
          k=1
          location=str(doc).split(',')
          data['username'].append(json_response['users'][i]['username'])
          data['name'].append(json_response['users'][i]['name'])
          data['tweet'].append(json_response['data'][i]['text'])
          data['location'].append(" ".join(location))
          break
      if k==0:
        data['username'].append(json_response['users'][i]['username'])
        data['name'].append(json_response['users'][i]['name'])
        data['tweet'].append(json_response['data'][i]['text'])
        data['location'].append("")      
    except:
      data['username'].append(json_response['users'][i]['username'])
      data['name'].append(json_response['users'][i]['name'])
      data['tweet'].append(json_response['data'][i]['text'])
      data['location'].append("")

  pr=pd.DataFrame(data)
  count=0
  for i in pr['location']:
    if i!='':
      count+=1
  return pr

def final_tweet_df(tweetDf):
  coor={'geometry':[],'address':[]}
  for i in tweetDf.location:
    if i=="":
      coor['geometry'].append(None)
      coor['address'].append(None)
    else:
      a=gpd.tools.geocode(i, provider='nominatim',timeout=100, user_agent="my-application")
      coor['geometry'].append(a['geometry'][0])
      coor['address'].append(a['address'][0])
      

  a=pd.DataFrame(coor)
  final_data=pd.concat([tweetDf,a],axis=1)
  return final_data

def get() :
  x=t.texts_to_sequences([title]) # pro
  x=sequence.pad_sequences(x,maxlen=250)
  p=model.predict(x)
  print(p)
  pred=np.argmax(p,axis=1)
  print(pred)
  if pred[0]==1:
    st.write('yes')
  else:
    st.write('no')

title = st.text_input('Text', '')
if st.button('Submit'):
     get()
st.title("Twitter Sentiment Analyser")


col1,col2 = st.columns(2)

option = col1.selectbox(
     'Select Brand Name',
     ('Iphone', 'Samsung', 'Xiaomi'))
col1.title("Brand Info")
num_tweets= col2.select_slider('Select number of tweets: ',options=['1', '2', '3', '4', '5', '6', '7'])
json_response=tweet_scrape(int(num_tweets))
tweetDf=tweet_df(json_response)
finalDf=final_tweet_df(tweetDf)
colx1,colx2,colx3,colx4=st.columns(4)

colx1.image("https://media.wired.com/photos/5bcea2642eea7906bba84c67/master/w_2560%2Cc_limit/iphonexr.jpg")
colx2.write("Name: option")
colx2.write("Model: option")
colx2.write("Color: option")
colx2.write("Price: option")
col2.title("Data Frame")
colx3.write(tweetDf[['name','tweet','location']])
coly1,coly2,coly3=st.columns(3)
chart_data = pd.DataFrame(
     np.random.randn(50, 2),
     columns=["a", "b"])


labels = 'Yes', 'No'
sizes = [70, 30]
explode = (0.1, 0)
coly1.write('Pie Chart')
fig1, ax1 = plt.subplots()
fig1.set_facecolor('#0D1117')
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90,textprops={'color':"w"})
ax1.axis('equal')

coly1.pyplot(fig1)
coly2.write('Bar Chart')
coly2.bar_chart(chart_data,width=500, height=540)
coly3.write('Map data')

d={'latitude':[],'longitude':[]}
for i in finalDf['geometry']:
  if str(i)[0]=='P':
    s=str(i).split('(')[1][:-1]
    p=s.split(" ")
    d['latitude'].append(float(p[1]))
    d['longitude'].append(float(p[0]))
data_of_map = pd.DataFrame(d)
coly3.map(data_of_map)      #-118.24544999999995 34.053570000000036
