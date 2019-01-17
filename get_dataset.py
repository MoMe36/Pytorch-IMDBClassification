import keras 
import numpy as np 
import pickle 

num_words = 3000
index_from = 3 

train,test = keras.datasets.imdb.load_data(num_words=num_words, index_from=index_from)
train_x, train_y = train 
test_x, test_y = test 

word2id = keras.datasets.imdb.get_word_index()  
word2id ={k:(v + index_from) for k, v in word2id.items()}

word2id['<PAD>'] = 0 
word2id['<START>'] = 1
word2id['<UNK>'] = 2 

id2word = {v:k for k,v in word2id.items()}

files = [train_x,train_y, test_x, test_y, word2id, id2word]
names = ['train_x', 'train_y', 'test_x', 'test_y', 'word2id', 'id2word']

for f,n in zip(files, names): 
    print('Saving ', n)
    pickle.dump(f, open('./imdb_dataset/{}'.format(n), 'wb'))