import tensorflow as tf
import numpy as np
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import words
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

nltk.download('words')

with open('content.json') as content:
  data1 = json.load(content)


tags = []
inputs = []
responses = {}

for intent in data1['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['input']:
    inputs.append(lines)
    tags.append(intent['tag'])

data = pd.DataFrame({"inputs":inputs,
                     "tags":tags})

#printing data
print(data)

data = data.sample(frac=1)

#removing punctuations
import string
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))
print()
print(data)

#tokenize the data
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])
#apply padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)
#encoding the outputs
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
print()
print(input_shape)
print()

#define vocabulary
vocabulary = len(tokenizer.word_index)
print("number of unique words : ",vocabulary)
output_length = le.classes_.shape[0]
print("output length: ",output_length)

#creating model
i = Input(shape = (input_shape,))
x = Embedding(vocabulary+1,10)(i)
x = Bidirectional(LSTM(10, return_sequences=True))(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(10, return_sequences=True, go_backwards=True))(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(output_length,activation="softmax")(x)
model = Model(i,x)

#compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',metrics=['accuracy'])

#training the model
train = model.fit(x_train,y_train,epochs=80,batch_size=8)

#plotting modal accuracy
plt.plot(train.history['accuracy'],label='training set accuracy')
plt.plot(train.history['loss'],label='training set loss')
plt.legend()


#prediction_input = input('You: ')


def chat(input):
  import random
  #testing and chatting to the bot
  print("Start Talking To Heron! (Type bye to stop)")
  while True:

    texts_p = []
    
    prediction_input = input
    #removing punctuation and converting to lowercase
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)
    #texts_p =  texts_p.split()
    
      
    #tokenizing and padding
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],input_shape)

    all_zero = np.all((prediction_input==0))
    if all_zero:
        return "I can't understand what you are saying. Can you please rephrase that?"
    else:
              

      #getting output from model
      output = model.predict(prediction_input)
      output = output.argmax()
              
                

      #finding the right tag and predicting 
      response_tag = le.inverse_transform([output])[0]
      response = random.choice(responses[response_tag])
                
      if response_tag == "goodbye":
          break
      return response
      
        
