import os
from tqdm import tqdm
import numpy as np 
import torch 
import pickle 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser 
import matplotlib.pyplot as plt 
# plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d




class Review(Dataset):

    def __init__(self, x_path, y_path, max_len = 350): 

        super().__init__()

        self.x = pickle.load(open(x_path, 'rb'))
        self.y = pickle.load(open(y_path, 'rb'))

        self.max_len = max_len
    def __len__(self): 

        return self.x.shape[0] 

    def __getitem__(self, ind): 

        x = self.x[ind][:self.max_len]

        padded = torch.zeros(self.max_len)
        x = torch.tensor(x).reshape(-1)
        padded[-x.shape[0]:] = x
        
        y = torch.tensor(self.y[ind]).float().reshape(-1)

        return padded.long(), y

def test_dataset(): 

    dataset = Review('./imdb_dataset/train_x', './imdb_dataset/train_y')
    x,y = dataset[898]

    return dataset

def test_loader(): 

    d = test_dataset()
    loader = DataLoader(d, batch_size = 10, shuffle = True)

    # for x,y in loader: 
    #     x = x.transpose(0,2)
    #     x = x.transpose(1,2)
    #     input(x.shape)

    return loader


class NLP(nn.Module):

    def __init__(self): 

        super().__init__()

        self.E = nn.Embedding(3000, 32)
        self.l1 = nn.Linear(32, 64)
        self.gru = nn.GRU(64,100,1)
        self.out = nn.Linear(100,1)


    def forward(self,x): 

        x = self.E(x)
        # print(x.shape)
        x = F.relu(self.l1(x))
        # print(x.shape)
        x = x.transpose(0,1)
        # print(x.shape)
        x, h = self.gru(x)
        # print(x.shape)
        x = x[-1]
        # print(x.shape)
        pred= torch.sigmoid(self.out(x))
        # input(pred.shape)
        return pred

    def pred_along_text(self, x): 

        x = self.E(x)
        # print(x.shape)
        x = F.relu(self.l1(x))
        # print(x.shape)
        x = x.transpose(0,1)
        # print(x.shape)
        x, h = self.gru(x)

        x = x.squeeze(1)

        preds = torch.sigmoid(self.out(x))
        return preds


def to_sentence(x): 

    text = ''
    for w in x: 
        if w != 0:
            text += '{} '.format(i2w[w])

    return text

def test_model():
    
    l = test_loader()
    nlp = NLP()

    # adam = optim.Adam(nlp.parameters())

    for x,y in l:
        # print(x.shape)
        preds = nlp(x)
        # input(y)
        # loss = F.binary_cross_entropy(preds, y)
        # print(loss)

# test_model()

def train(model, adam, loader, nb_batch): 

    epoch_loss = 0.
    bar = tqdm(total = nb_batch)
    for i, (x,y) in enumerate(loader):

        preds = nlp(x)
        loss = F.binary_cross_entropy(preds, y)
        adam.zero_grad()

        loss.backward()
        adam.step()

        epoch_loss += loss.item()
        bar.update(1)
        bar.set_description('Loss: {:.4f}'.format(epoch_loss/float(i+1)))

    return epoch_loss

def make_imgs(preds, label, text, selected_texte, ind): 

    path = './videos/{}/'.format(ind)
    try: 
        os.makedirs(path)
    except: 
        pass

    color = ((0.9,0.5,0.1))
    f, ax = plt.subplots(2,1)
    text_list = text.split()

    length = [len(preds), len(text_list)]

    for i in range(min(length)):
        
        ax[0].clear()
        ax[1].clear()

        current_text = ''
        for idx in range(i):
            current_text += '{} '.format(text_list[idx]) 

        ax[0].plot(preds[:i], color = color, alpha = 0.2)
        ax[0].plot(gaussian_filter1d(preds[:i], sigma = 5), color = color)
        ax[0].set_xlim(0, len(preds))
        ax[0].set_ylim(0, 1.05)

        ax[0].set_title('Pred: {:.2f} -- Label : {}'.format(preds[i], label))
        ax[1].patch.set_visible(False)
        ax[1].axis('off')

        ax[1].text(0.0,0.5, current_text, wrap = True, family = 'serif', va = 'top')
        plt.savefig('{}{}.png'.format(path, i))
        # plt.pause(0.1)

def compute_preds(x, model): 

    selected = torch.where(x > 0, torch.arange(x.shape[1]), x)
    stop_pad = torch.argmin(selected)

    new_x = x[0,stop_pad:].reshape(1,-1)

    with torch.no_grad():
        new_preds = model.pred_along_text(new_x)

    new_preds = new_preds.numpy().reshape(-1).tolist()

    return new_preds

def get_subtext(x, nb_words = 10):

    text = to_sentence(x.numpy().reshape(-1).tolist()) 

    text_list = text.split()

    margin = int(float(len(text_list))/nb_words)

    pos = np.arange(1, len(text_list), margin).astype(int).tolist()
    pos[-1] = len(text_list) -1
    selected_texte = [text_list[i] for i in pos]

    return text, text_list, selected_texte


def create_imgs(x,y, model, i): 


    preds = compute_preds(x, model)
    text, text_list, selected_texte = get_subtext(x)
    make_imgs(preds, y.reshape(-1).item(), text, selected_texte, i)

def test(model, loader): 

    color = ((0.9,0.5,0.1))
    c2 = ((1.,0.,0.))

    for i, (x,y) in enumerate(loader): 

       
        new_preds = compute_preds(x, model)
        text, text_list, selected_texte = get_subtext(x)
        # text = to_sentence(x.numpy().reshape(-1).tolist()) 

        # text_list = text.split()
        
        # margin = int(float(len(new_preds))/10)
        # pos = np.arange(1, len(text_list), margin).astype(int).tolist()
        # pos[-1] = len(text_list) -1
        # selected_texte = [text_list[i] for i in pos]

        # ax[1].text(0.1,0.5, text, wrap = True)
        # make_imgs(new_preds, y.reshape(-1).item(), text, selected_texte)
        
        # input(selected_texte)
        plt.cla()
        plt.ylim(0.,1.05)
        plt.ylabel('Prediction')
        plt.xlabel('Rewiew progress')
        plt.title('Ex: {} Pred: {:.3f} -- Label: {}'.format(i, new_preds[-1], y.reshape(-1).item()))
        # plt.plot(preds, color = color, alpha = 0.4)
        # plt.plot(gaussian_filter1d(preds, sigma = 4), color = color)
        plt.xticks(np.arange(len(selected_texte))*float(len(new_preds))/float(len(selected_texte)),  selected_texte)
        plt.plot(new_preds, color = c2, alpha = 0.4)
        plt.plot(gaussian_filter1d(new_preds, sigma = 4), color = c2)


        plt.pause(0.1)


        print(text)
        with torch.no_grad(): 
            pred = model(x)
        input('EX: {} Pred: {:.4f} -- Label: {}\n\n\n'.format(i, pred.reshape(-1).item(), y.reshape(-1).item()))


parser = ArgumentParser()
parser.add_argument('--eval', action =  "store_true")
parser.add_argument('--acc', action = "store_true")
args = parser.parse_args() 

i2w = np.load('./imdb_dataset/id2word')

dataset = Review('./imdb_dataset/train_x', './imdb_dataset/train_y')

nlp = NLP()
try: 
    nlp.load_state_dict(torch.load('./nlp.pt'))
except: 
    pass
if(args.eval): 

    dataset = Review('./imdb_dataset/test_x', './imdb_dataset/test_y')

    if args.acc: 
        loader = DataLoader(dataset, batch_size = len(dataset), shuffle = False)

        for x,y in loader: 
            with torch.no_grad():
                preds = nlp(x)

            o, z = torch.ones_like(preds), torch.zeros_like(preds)
            preds = torch.where(preds > 0.5, o, z)

            preds = preds.reshape(-1).numpy().tolist()
            y = y.reshape(-1).numpy().tolist()

            corrects = 0
            for pp, yy in zip(preds, y): 
                if pp == yy: 
                    corrects += 1

            print('Accuracy: {}/{} - > {:.3f}%'.format(corrects, len(dataset), corrects/float(len(dataset))*100.))



    loader = DataLoader(dataset, batch_size = 1, shuffle = False)
    # test(nlp, loader)
    inds = [59]

    for i in inds: 
        x,y = loader.dataset[i]
        create_imgs(x.unsqueeze(0),y, nlp, i)


else: 
    
    loader = DataLoader(dataset, batch_size = 32, shuffle = True)
    batchs_per_epochs = len(dataset) // loader.batch_size


    adam = optim.Adam(nlp.parameters(), lr = 1e-3)

    epochs = 10
    for epoch in tqdm(range(epochs)): 

        torch.save(nlp.state_dict(), 'nlp.pt')
        epoch_loss = train(nlp, adam, loader, batchs_per_epochs)
        print(epoch_loss)






