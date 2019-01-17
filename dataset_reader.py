import numpy as np 

i2w = np.load('./imdb_dataset/id2word')

train = np.load('./imdb_dataset/train_x')
labels = np.load('./imdb_dataset/train_y') 

classes = ['Negative', 'Positive'] 

for i in range(train.shape[0]): 

	x = train[i]
	s = '' 
	for w in x: 
		s += i2w[w] + ' '
	
	print(s) 
	print('Result: {}'.format(classes[labels[i]]))
	input()
