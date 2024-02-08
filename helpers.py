import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout, Embedding
import matplotlib.pyplot as plt
import seaborn

import tensorboard
from datetime import datetime
from tqdm.auto import tqdm

def pre_process_input_data(filepath='./data/concept_net/tweets.csv',encoding='cp1252',num_samples=None,random_state=None): #Change encoding if not on windows
    tweets = pd.read_csv(filepath,encoding=encoding,header=None)
    tweets.columns = ['target','id','date','flag','username','text'] #Change column names to things that make sense
    tweets = tweets.drop(columns=['id','date','flag','username']) #Remove unneeded columns from memory

    tweets = tweets.replace({'target':{0:0,4:1}}) #Dataset has only 0=negative sent, 4=positive sent, remappping to 0,1 respectivly
    if num_samples:
        tweets = tweets.groupby('target').sample(num_samples,random_state=random_state)

    return tweets

def load_lexicon(filename): #Stolen from Dr. Larson
    """
    Load a file from Bing Liu's sentiment lexicon
    (https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html), containing
    English words in Latin-1 encoding.
    
    One file contains a list of positive words, and the other contains
    a list of negative words. The files contain comment lines starting
    with ';' and blank lines, which should be skipped.
    """
    lexicon = []
    with open(filename, encoding='latin-1') as infile:
        for line in infile:
            line = line.rstrip()
            if line and not line.startswith(';'):
                lexicon.append(line)
    return lexicon
  
def load_embeddings(filename): #Stolen from Dr.Larson
    """
    Load a DataFrame from the generalized text format used by word2vec, GloVe,
    fastText, and ConceptNet Numberbatch. The main point where they differ is
    whether there is an initial line with the dimensions of the matrix.
    """
    labels = []
    rows = []
    with open(filename, encoding='utf-8') as infile:
        for i, line in tqdm(enumerate(infile),total=2196017):
            items = line.rstrip().split(' ')
            if len(items) == 2:
                # This is a header row giving the shape of the matrix
                continue
            labels.append(items[0])
            values = np.array([float(x) for x in items[1:]], 'f')
            rows.append(values)
    
    arr = np.vstack(rows)
    return pd.DataFrame(arr, index=labels, dtype='f')

def generatSimpleDenseNetwork(return_callbacks = True):
    model = Sequential()
    model.add(tf.keras.Input(shape=(300,)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
    )
    callbacks = []
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    callbacks.append(tensorboard_callback)

    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001)) #Early stop
    if return_callbacks:
        return model, callbacks
    else:
        return model
    
def generateCNN(return_callbacks = True):
    #ensure the input is the 300 dim vector
    sequence_input = Input(shape=(300,1))
    x = Conv1D(128, 5, activation='relu', kernel_initializer='he_uniform')(sequence_input)
    x = MaxPooling1D(5)(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    preds = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(x)

    model = Model(sequence_input, preds)
    callbacksCNN = []
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5,)
    model.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['acc'])
    
    callbacksCNN.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001)) #Early stop
    if return_callbacks:
        return model, callbacksCNN
    else:
        return model
    
def generatSimpleRecurrentNetwork(return_callbacks = True):
    RNN_STATESIZE = 100

    rnns = []
    input_holder = tf.keras.Input(shape=(300,1))
    x = layers.SimpleRNN(RNN_STATESIZE, dropout=0.2, recurrent_dropout=0.2)(input_holder)
    #use a different activation function
    x = layers.Dense(1, activation='relu')(x)
    simple_RNN = Model(inputs=input_holder,outputs=x)

    opt = Adam(lr=0.0001, epsilon=0.0001, clipnorm=1.0)

    callbacks = []
    simple_RNN.compile(loss='binary_crossentropy', 
              optimizer= opt, 
              metrics=['accuracy'])

    #logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # callbacks.append(tensorboard_callback)

    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001)) #Early stop
    if return_callbacks:
        return simple_RNN, callbacks
    else:
        return simple_RNN

def plot_model_stats(mdl1,mdl2,mdl1_name='Bert Classifier',mdl2_name = 'Glove Classifier'):
    fig, axs = plt.subplots(2,2,sharex=True,figsize=(15,7))
    axs[0,0].plot(mdl1.history['accuracy'])
    axs[0,0].plot(mdl1.history['val_accuracy'])
    axs[0,0].set_title(f'{mdl1_name} Accuracy')
    axs[0,0].legend(['train','validation'],loc='lower right')
    axs[0,0].sharey(axs[0,1])

    axs[0,1].plot(mdl2.history['accuracy'])
    axs[0,1].plot(mdl2.history['val_accuracy'])
    axs[0,1].set_title(f'{mdl2_name} Accuracy')
    axs[0,1].legend(['train','validation'],loc='lower right')
    
    
    axs[1,0].sharey(axs[1,1])
    axs[1,0].plot(mdl1.history['loss'])
    axs[1,0].plot(mdl1.history['val_loss'])
    axs[1,0].set_title(f'{mdl1_name} Loss')
    axs[1,0].legend(['train','validation'],loc='upper right')
    
    axs[1,1].plot(mdl2.history['loss'])
    axs[1,1].plot(mdl2.history['val_loss'])
    axs[1,1].set_title(f'{mdl2_name} Loss')
    axs[1,1].legend(['train','validation'],loc='upper right')

    for ax in axs.flat:
        ax.grid()
        ax.set(xlabel='Epoch')

def plot_test_hists(diff_1,diff_2,name_1,name_2,figsize=(10,5)):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=figsize,sharey=True)
    counts_1,bins_1 = np.histogram(diff_1,10)
    counts_2,bins_2 = np.histogram(diff_2,10)
    ax1.hist(bins_1[:-1],bins_1,weights=counts_1)
    ax2.hist(bins_2[:-1],bins_2,weights=counts_2)

    ax1.set_title(name_1)
    ax2.set_title(name_2)
    ax1.grid()
    ax2.grid()

def result_plotter(data,x,y1,y2,y_lim=[0,1],y1_name='Bert',y2_name='Glove'):
    fig,axs = plt.subplots(2,2,figsize=(15,5),layout='tight')

    # plt.figure(figsize=(15,5))
    # plt.subplot(121)
    seaborn.swarmplot(x=x, y=y1, data=data,ax=axs[0,0])
    seaborn.barplot(x=x, y=y1, data=data, capsize=.1,ax=axs[0,1])
    axs[0,0].set(title=f'{y1_name} Swarm Plot')
    axs[0,1].set(title=f'{y1_name} Bar Plot')

    seaborn.swarmplot(x=x, y=y2, data=data,ax=axs[1,0])
    seaborn.barplot(x=x, y=y2, data=data, capsize=.1,ax=axs[1,1])
    for ax in axs.flat:
        ax.set(ylim=y_lim)
        ax.set(xlabel='Group')
    axs[0,0].set(xlabel='')
    axs[0,1].set(xlabel='')
    
    axs[1,1].set(ylabel='')
    axs[0,1].set(ylabel='')

    axs[0,0].set(ylabel=f'Sentiment Score')
    axs[1,0].set(ylabel='Sentiment Score')
    
    axs[1,0].set(title=f'{y2_name} Swarm Plot')
    axs[1,1].set(title=f'{y2_name} Bar Plot')