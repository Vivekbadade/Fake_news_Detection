# import numpy as np # linear algebra
# import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd


# Sample news articles

data = pd.read_csv(r"news.csv")
data = data.dropna()
data = data.reset_index(drop=True)
data['label']=data['label'].apply( lambda x: 1 if x=='REAL' else 0)

# Tokenize the news articles
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(data['text']))

# Convert news articles to sequences
sequences = tokenizer.texts_to_sequences(list(data['text']))

# Pad the sequences to have the same length
padded_sequences = pad_sequences(sequences)             

# Build the model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                           output_dim=32,
                           input_length=max(len(seq) for seq in padded_sequences)),
    keras.layers.LSTM(16),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(padded_sequences, data['label'], epochs=5)

#save model
model.save('LSTM_news_detection_model.h5')

fake_news = "SHOCK VIDEO : Hillary Needs Help Climbing ONE"
fake_news_sequence = tokenizer.texts_to_sequences([fake_news])
padded_fake_news_sequence = pad_sequences(fake_news_sequence, maxlen=padded_sequences.shape[1])
prediction = model.predict(padded_fake_news_sequence)

# Print the result
if prediction < 0.5:
    print("Real News")
else:
    print("Fake News")

