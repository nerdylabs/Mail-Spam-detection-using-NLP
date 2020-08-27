import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

print('\n Done Importing\n')

####Hyperparameters####
VOCAB_SIZE = 30000
EMBEDDING_DIM = 16
EPOCHS = 15
MAX_LEN = 32
TRUNC_TYPE = 'post'
PADD_TYPE = 'post'
UNK_TOK = '<UNK>'
TRAINING_SIZE = 1000

print('\nHyperparameters Done\n')

dataset = pd.read_csv('Spam dataset.csv')
print('dataset: ')

print(dataset.head(), '\n')
print('Null values in the dataset: ')

print(dataset.isnull(), '\n')
print('Sum of Null values in each column: ')

print(dataset.isnull().sum(), '\n')

dataset["Category"] = [1 if each == "spam" else 0 for each in dataset["Category"]]

print('dataset: ')
print(dataset.head(), '\n')

X = dataset.iloc[:, 1].values
X = X.reshape(-1, 1)
print('Type of X:', type(X))
print('shape of X: ', X.shape)
#print(X)

Y = dataset.iloc[:, 0].values
y = Y.reshape(-1, 1)
print('Type of Y; ', type(Y))
print('shape of Y: ', y.shape)
#print(y)
X = X.tolist()
y = y.tolist()
print('Length of X: ', len(X), '\n')

####Train test Split####
Training_Sentences = X[TRAINING_SIZE:]
Training_Labels = y[TRAINING_SIZE:]
print('Length of training samples ', len(Training_Sentences), '\n')

Testing_Sentences = X[0: TRAINING_SIZE]
Testing_Lables = y[0:TRAINING_SIZE]
print('Length of testing samples', len(Testing_Sentences), '\n')


####TOKENIZING THE WORDS#####
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=UNK_TOK)
tokenizer.fit_on_texts(Training_Sentences)

word_index = tokenizer.word_index

#print(word_index, '\n \n \n')

Training_Sequences = tokenizer.texts_to_sequences(Training_Sentences)
Training_pad = pad_sequences(Training_Sequences, maxlen=MAX_LEN, padding=PADD_TYPE, truncating=TRUNC_TYPE)

Testing_Sequences = tokenizer.texts_to_sequences(Testing_Sentences)
Testing_pad = pad_sequences(Testing_Sequences, maxlen=MAX_LEN, padding=PADD_TYPE, truncating=TRUNC_TYPE)
#####BUILDING THE MODEL############

model = tf.keras.Sequential()
print('#####MODEL INSTANCE Done#####')
model.add(tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(24, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

print('#########################')
print(model.summary())
print('#########################')

########Converting to numpy array##########
Training_Sequences_padded = np.asarray(Training_pad)
Testing_Sequences_padded = np.asarray(Testing_pad)
Training_Labels = np.asarray(Training_Labels)
Testing_Lables = np.asarray(Testing_Lables)

######Training THE MODEL###################
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(Training_Sequences_padded, Training_Labels, validation_data=(Testing_Sequences_padded, Testing_Lables), epochs=EPOCHS)

def Plot(history, string):
	plt.plot(history.history[string])
	plt.plot(history.history['val_' + string])
	plt.xlabel('EPOCHS')
	plt.ylabel(string)
	plt.legend([string, 'val_' + string])
	plt.show()

Plot(history, "accuracy")
Plot(history, "loss")


print("######DONE########")

##########SAVING THE MODEL#############
model.save('Spam Model1')

text = 'Get 100% off for this random thing that you won in a lottery!!!!!'
print(text)
Test = tokenizer.texts_to_sequences([text])[0]
Test_padded = pad_sequences([Test], maxlen=MAX_LEN, padding=PADD_TYPE, truncating=TRUNC_TYPE)
Test_padded = np.asarray(Test_padded)
print('Done')
print(Test_padded.shape)

ypred = model.predict(Test_padded)

print('output class: ', ypred)

