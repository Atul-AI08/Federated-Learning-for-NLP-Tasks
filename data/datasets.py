"""Module for datasets classes"""
import pandas as pd 
from torch import tensor
from torch.utils.data import Dataset
from utils.vocab import Vocabulary 

# Dataset classes
class TweetSentimentDataset(Dataset):
    """Dataset Class for TwitterSentiment140 dataset
    
    params
    ---------------
    + csv_path : file path to the dataset
        
    + freq_threshold : Used during constructing vocabulary. Defaults to 5 
    
    + vocab_file : If given vocabulary will be loaded from this file
    
    """
    def __init__(self,csv_path,num_words=100,freq_threshold = 5,vocab_file=None,create_vocab=True,dataidxs=None):
        self.df = pd.read_csv(csv_path)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True,inplace=True)
        self.df['target'].replace({4:1},inplace=True)
  
        self.num_words = num_words
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
        
    + freq_threshold : Used during constructing vocabulary. Defaults to 5 
    
    + vocab_file : If given vocabulary will be loaded from this file
    
    """
    def __init__(self,csv_path,num_words=100,freq_threshold = 5,vocab_file=None,create_vocab=True,dataidxs=None):
        self.df = pd.read_csv(csv_path)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True,inplace=True)

        self.df['target'].replace({'positive':1},inplace=True)
        self.df['target'].replace({'negative':0},inplace=True)
        
        self.num_words = num_words
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