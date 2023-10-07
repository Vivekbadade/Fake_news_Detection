import pandas as pd
import numpy as np
import streamlit as st
# import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
# from keras_preprocessing.sequence import pad_sequences

loaded_model = keras.models.load_model(r'E:\New folder (3)\EDI\streamlit\LSTM_news_detection_model.h5')
data=pd.read_csv(r'E:\New folder (3)\EDI\streamlit\news.csv')
# Tokenize the news articles
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(data['text']))

# Convert news articles to sequences
sequences = tokenizer.texts_to_sequences(list(data['text']))

# Pad the sequences to have the same length
padded_sequences = pad_sequences(sequences)

st.title("Fake News Detection System")
def fakenews():
    fake_news = st.text_area("Enter Any News Headline: ")
    if len(fake_news) <= 1:
        st.write("  ")
    else:
        fake_news_sequence = tokenizer.texts_to_sequences([fake_news])
        padded_fake_news_sequence = pad_sequences(fake_news_sequence, maxlen=padded_sequences.shape[1])
        prediction = loaded_model.predict(padded_fake_news_sequence)
        if prediction > 0.5:
            st.title("Real News ✔️")
        else:
            st.title("Fake News ❌")
        # st.title(prediction)
if __name__ == '__main__':
    fakenews()
