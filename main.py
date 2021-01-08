import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import string
import nltk
import emoji
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk import word_tokenize
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

vocab_size = 10000
maxlen = 50
output_dim = 32
stopword = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
all_words = []
emoji_reg = emoji.get_emoji_regexp()

def read_file(name):
    """
  Reads csv file and keeps row with acceptables class values
  """
    df = pd.read_csv(name)
    df = df[df['Class'].isin(['0', '1'])]
    return df.dropna()


def split_data(x, y):
    """
  Split into train test
  """
    y_data = np.array([int(i) for i in y])
    return train_test_split(x_data, y_data, test_size=0.3)


def sanitize_text(text):
    """
  Cleans tweet data by removing mentions, http links and retweet tags
  """
    try:
        # replace retweet tags
        text = re.sub(r'^(RT @\w+)', '', text)
        # replace website URLs
        text = re.sub(r'(https://)([A-Za-z0-9./]+)', '', text)
        # replace mentions
        text = re.sub(r'(@\w+)', '', text)
        # replace emojis
        text = emoji_reg.sub(u'', text)

        text_c = re.sub('[0-9]+', '', text)
        text_pc = "".join([word.lower() for word in text_c if word not in string.punctuation])
        tokens = word_tokenize(text_pc)
        text = [ps.stem(word) for word in tokens if word not in stopword]
        all_words.extend(text)
    except Exception as e:
        print(e)
    return text


def clean_data(df):
    """
  Creates new column with cleaned text and keeps only top 10000 most frequent words
  """
    df['cleaned_text'] = df['Text'].apply(sanitize_text)
    keys = dict(collections.Counter(all_words).most_common(vocab_size)).keys()

    def keep_top(text):
        text = [i for i in text if i in keys]
        return " ".join(text)

    df['cleaned_text'] = df['cleaned_text'].apply(keep_top)
    return df


def encode_data(df, x):
    """
  Integer encodes words in text to prepare data for embeddings
  """
    encod_corp = []
    for i, doc in enumerate(df[x].tolist()):
        encod_corp.append(one_hot(doc, vocab_size))
    pad_corp = pad_sequences(encod_corp, maxlen=maxlen, padding='pre', value=0.0)
    return pad_corp


def create_model():
    """
  Create model as mentioned in image
  """
    model = Sequential()
    model.add(layers.Input(shape=(maxlen,), dtype='float64'))
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=maxlen))

    model.add(layers.Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(50, 32)))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Bidirectional(layers.LSTM(256)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))

    adam = Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def run_and_evaluate(epochs=1, batchsize=512):
    """
  trains model and evaluates on given metrics
  """
    model.fit(x_train, y_train, epochs=epochs, batch_size=batchsize, verbose=1)
    y_pred = model.predict(x_test)
    y_pred = [round(i[0]) for i in y_pred]
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {acc}")
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"Maththews correlation coefficient: {mcc}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")


if __name__ == "__main__":
    print("reading dataset...")
    df = read_file("data/CVAssignmentDataset.csv")
    print("data read, cleaning tweets")
    df_cleaned = clean_data(df)
    x_data = encode_data(df_cleaned, 'cleaned_text')
    x_train, x_test, y_train, y_test = split_data(x_data, df_cleaned['Class'].tolist())
    print("created train test splits, starting model training and evaluation")
    model = create_model()
    run_and_evaluate(12, 512)
    print("done")
