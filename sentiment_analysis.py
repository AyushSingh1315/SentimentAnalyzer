
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense, Dropout,Embedding, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[3]:
'''Loading the data'''

#DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_COLUMNS = ["no.",'id','what','text']
DATASET_ENCODING = "ISO-8859-1"
# targets =  list(pd.read_csv("D:training_data/sentiment140/training.1600000.processed.noemoticon.csv",encoding =DATASET_ENCODING , names=DATASET_COLUMNS)['target'])
# text =  list(pd.read_csv("D:training_data/sentiment140/training.1600000.processed.noemoticon.csv",encoding =DATASET_ENCODING , names=DATASET_COLUMNS)['text'])
sentiment = list(pd.read_csv("D:training_data/twitter-sentiment/Sentiment Analysis Dataset 2.csv",encoding=DATASET_ENCODING,names=DATASET_COLUMNS)['id'])
text = list(pd.read_csv("D:training_data/twitter-sentiment/Sentiment Analysis Dataset 2.csv",encoding=DATASET_ENCODING,names=DATASET_COLUMNS)['text'])


# In[4]:
'''Using only a part of the loaded data'''

sentiments=[]
texts=[]
num_zeros=0
num_ones=0
for i in range(len(sentiment)):
    if num_zeros==100000 and num_ones==100000:
        break
    
    if sentiment[i]==0:
            if num_zeros<=100000:
                sentiments.append([1,0])
                texts.append(text[i])
                num_zeros+=1
    else:
            if num_ones<=100000:
                sentiments.append([0,1])
                texts.append(text[i])
                num_ones+=1
                
                
      


# In[5]:


sentiments=sentiments[1:]
texts = texts[1:]
text=[]
sentiment=[]


# In[30]:


np.save('sentiments.npy',sentiments)
np.save('texts.npy',texts)

len(texts[4])


# In[7]:


max_feature = 2000
a = Tokenizer(num_words=max_feature,split=' ')       
a.fit_on_texts(texts)
# print(a.word_counts)
# print(a.document_count)
# print(a.word_index)
# print(a.word_docs)
encoded_docs = a.texts_to_sequences(texts)
encoded_docs = pad_sequences(encoded_docs)
print(encoded_docs)


# In[10]:


encoded_docs = np.array(encoded_docs)
encoded_docs = np.reshape(encoded_docs,(200001,101))
sentiments=np.array(sentiments)
sentiments=np.reshape(sentiments,(200001,2))


# In[11]:


print(np.shape(encoded_docs),np.shape(sentiments))




# In[38]:

'''Building the model'''
model =Sequential()

model.add(Embedding(max_feature, 64,input_length=101))
model.add(LSTM(32))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(2,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.optimizer.lr = 0.05


# In[39]:


model.fit(encoded_docs,sentiments,batch_size=3000,epochs=7)


# In[31]:


## SAVING THE MODEL
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 


# In[ ]:


# # load json and create model
# from tensorflow.keras.models import model_from_json
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")


# In[23]:


def predict_sentiment(my_text):
    my_text = [my_text]
    encoded_text = a.texts_to_sequences(my_text)
    encoded_text = pad_sequences(encoded_text,maxlen=101,value=0,dtype='int32')
    prediction = model.predict(encoded_text)
   
    if np.argmax(prediction)==1:
        print('Positive')
    else:
        print('Negative')
    print(prediction)    
        
    


# In[24]:


predict_sentiment("hell no")


# In[25]:


predict_sentiment("hell yeah")


# In[26]:


predict_sentiment("that idea sounds cool to me")


# In[27]:


predict_sentiment("I don't like this plan")


# In[28]:


predict_sentiment('what the hell is wrong with you')


# In[29]:


predict_sentiment('we are good')

