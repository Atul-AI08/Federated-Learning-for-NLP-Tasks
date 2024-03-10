"""Module for Vocabulary"""
from collections import Counter
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import json

class Vocabulary:
    """Class for dealing with constructing vocabulary for text dataset
    """
    def __init__(self,freq_threshold):
        self.freq_threshold = freq_threshold
        self.idx2str = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        self.str2idx = {value : key for key,value in self.idx2str.items()}
    
    def __len__(self):
        return len(self.idx2str)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.lower() for tok in word_tokenize(text) if tok.strip()]
    
    def build_vocabulary(self,sentence_list,save_to_file=None):
        """Builds dictionary for given sentences"""
        frequencies = Counter()
        start_idx = len(self.idx2str)
        for sentence in tqdm(sentence_list,desc="Building vocabulary",total=len(sentence_list)):
            # for word in self.tokenizer_eng(sentence):
            #     frequencies[word] += 1
            #     if frequencies[word]==self.freq_threshold:
            #         self.str2idx[word] = start_idx 
            #         self.idx2str[start_idx] = word
            #         idx += 1
            frequencies.update(Counter(self.tokenizer_eng(sentence)))
        updated = (word for word,count in frequencies.items() if count>=self.freq_threshold)
  
        self.str2idx.update({word:idx for idx,word in enumerate(updated,start_idx)})
        self.idx2str.update({idx:word for idx,word in enumerate(updated,start_idx)})            
      
        
        if save_to_file is not None:
            self.__save_vocab(save_to_file)
            print(f"\n[DEBUG] vocabulary saved in {save_to_file}\n") 
      
        
    def numericalize(self,text):
        """Assigns a number to each word in text and returns the list of those numbers"""
        tokens = self.tokenizer_eng(text)
        return [self.str2idx.get(word,self.str2idx["<UNK>"]) for word in tokens]
    
    def build_vocabulary_from_file(self,file):
        self.str2idx = self.__load_vocab(file)
        
    
    def __load_vocab(self,file):
        with open(file,'r') as f:
           return json.load(f)
       
    def __save_vocab(self,file):
        with open(file,'w') as f:
            json.dump(self.str2idx,f)
    