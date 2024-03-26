import re

import pandas as pd
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
new_stop_words = ['httpskeptical','go','undecided','annoy']

tweet_tokenizer = TweetTokenizer(preserve_case=False,reduce_len=True,strip_handles=True)

word_net_lemmatize = WordNetLemmatizer()

clean_regex = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
negations_dict = {
            "isn't":"is not", "aren't":"are not", "wasn't":"was not",
            "weren't": "were not", "haven't":"have not","hasn't":"has not",
            "hadn't":"had not","won't":"will not", "wouldn't":"would not",
            "don't":"do not", "doesn't":"does not", "didn't":"did not",
            "can't":"can not","couldn't":"could not",
            "shouldn't":"should not","mightn't":"might not",
            "mustn't":"must not"
            }

# Functions for text preprocessing

def convert_to_lower(text):
    return text.lower()

def remove_stop_words(text):
    return " ".join([t for t in text.split() if t not in stop_words])

def tokenize(text):
    return ' '.join(tweet_tokenizer.tokenize(text))

def lemmatize(text):
    tokens = [word_net_lemmatize.lemmatize(word,pos='v') for word in text.split() if word]
    tokens = [word_net_lemmatize.lemmatize(word,pos='n') for word in tokens]
    return " ".join(tokens) 

def clean_text(text):
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dict.keys()) + r')\b')

    text = convert_to_lower(text)
    text = re.sub(clean_regex, ' ', text)
    text = neg_pattern.sub(lambda x: negations_dict[x.group()], text)
    
    text = remove_stop_words(text)
    text = lemmatize(text)
    
    return text

# Function to load data
def load_data(csv_path, min_tweets = 20):
    columns = ["target", "ids", "date", "flag", "user", "text"]
    data = pd.read_csv(csv_path, encoding = "ISO-8859-1", header = None, names = columns)
    data.drop(["flag", "ids"], axis = 1, inplace = True)
    users = data.groupby(by = "user").apply(len) > min_tweets
    data = data[data.user.isin(users[users].index)]
    data["cleaned_text"] = data.text.apply(lambda x: clean_text(x))
    data.drop_duplicates(subset = ["cleaned_text"], keep = False, inplace = True)
    
    return data

if __name__ == '__main__':
    df = load_data("training.1600000.processed.noemoticon.csv")
    df.to_csv("cleaned_data.csv", index=False)