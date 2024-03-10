"""Module for datasets classes"""
import pandas as pd 
from torch import tensor
from torch.utils.data import Dataset
from utils.vocab import Vocabulary 

class TweetSentimentDataset(Dataset):
    """Dataset Class for TwitterSentiment140 dataset
    
    params
    ---------------
    + csv_path : file path to the dataset
    
    + transform : any transform function for applying on each sample before dispatch
    
    + freq_threshold : Used during constructing vocabulary. Defaults to 5 
    
    + vocab_file : If given vocabulary will be loaded from this file
    
    """
    def __init__(self,csv_path,num_words=100,freq_threshold = 5,vocab_file=None,create_vocab=True,dataidxs=None):
        self.df = pd.read_csv(csv_path)
        # print(self.df['text'].isnull().any().sum())
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True,inplace=True)
        
        #!Really necessary??? Should have taken care while building the dataset itself lol. 
        #?Dataset containes 4 for positive and 0 for negative.
        #Categorical crossentropy requires class indices as targets for example if target classes = ['negative','positive']
        #then the targets for computing loss are 0 and 1 respectively.
        self.df['target'].replace({4:1},inplace=True)
  
        self.num_words = num_words
        self.df = self.df.reset_index(drop=True)
        # print(self.df)
        self.data = self.df['cleaned_text']
        self.target = self.df['target']
        self.dataidxs = dataidxs

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.target = self.target[self.dataidxs]
            self.data = self.data.reset_index(drop=True)
            self.target = self.target.reset_index(drop=True)
            
        self.vocab = Vocabulary(freq_threshold)
        if create_vocab:
            self.vocab.build_vocabulary(self.data.tolist(),vocab_file)
        else:
            self.vocab.build_vocabulary_from_file(vocab_file)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        tweet = self.data[idx]
        target = self.target[idx]
            
        # numericalized_data = [self.vocab.str2idx["<SOS>"]]
        numericalized_data = self.vocab.numericalize(tweet)
        # numericalized_data.append(self.vocab.str2idx["<EOS>"])
        numericalized_data.extend([self.vocab.str2idx["<PAD>"] for _ in range(len(numericalized_data),self.num_words)])
        if len(numericalized_data)>self.num_words:
            numericalized_data = numericalized_data[:self.num_words]
        return tensor(numericalized_data),tensor(target)

class ImdbSentimentDataset(Dataset):
    """Dataset Class for Large Movie Review Dataset
    
    params
    ---------------
    + csv_path : file path to the dataset
    
    + transform : any transform function for applying on each sample before dispatch
    
    + freq_threshold : Used during constructing vocabulary. Defaults to 5 
    
    + vocab_file : If given vocabulary will be loaded from this file
    
    """
    def __init__(self,csv_path,num_words=100,freq_threshold = 5,vocab_file=None,create_vocab=True,dataidxs=None):
        self.df = pd.read_csv(csv_path)
        # print(self.df['text'].isnull().any().sum())
        self.df = self.df.iloc[:4000,:]
       
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True,inplace=True)
        
        #!Really necessary??? Should have taken care while building the dataset itself lol. 
        #?Dataset containes 4 for positive and 0 for negative.
        #Categorical crossentropy requires class indices as targets for example if target classes = ['negative','positive']
        #then the targets for computing loss are 0 and 1 respectively.
        self.df['sentiment'].replace({'pos':1},inplace=True)
        self.df['sentiment'].replace({'neg':0},inplace=True)
        # self.df['sentiment'] = self.df['sentiment'].str.replace('pos',1)
        # self.df['sentiment'] = self.df['sentiment'].str.replace('neg',0)
        
        self.num_words = num_words
        self.df = self.df.sample(frac=1,random_state=42).reset_index(drop=True)
        # print(self.df)
        self.data = self.df['text']
        self.target = self.df['sentiment']
        self.dataidxs = dataidxs
        
        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.target = self.target[self.dataidxs]
            self.data = self.data.reset_index(drop=True)
            self.target = self.target.reset_index(drop=True)
            
        self.vocab = Vocabulary(freq_threshold)
        if create_vocab:
            self.vocab.build_vocabulary(self.data.tolist(),vocab_file)
        else:
            self.vocab.build_vocabulary_from_file(vocab_file)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        tweet = self.data[idx]
        target = self.target[idx]
        
        if self.transform:
            tweet,target = self.transform((tweet,target))
            
        # numericalized_data = [self.vocab.str2idx["<SOS>"]]
        numericalized_data = self.vocab.numericalize(tweet)
        # numericalized_data.append(self.vocab.str2idx["<EOS>"])
        numericalized_data.extend([self.vocab.str2idx["<PAD>"] for _ in range(len(numericalized_data),self.num_words)])
        if len(numericalized_data)>self.num_words:
            numericalized_data = numericalized_data[:self.num_words]
        return tensor(numericalized_data),tensor(target).float()