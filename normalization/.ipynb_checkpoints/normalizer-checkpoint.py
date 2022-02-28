import urllib.request
import gensim
from gensim.models import word2vec

import nltk.data
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, RegexpTokenizer 

import pandas as pd
from tqdm.notebook import tqdm 

import pymorphy2

tqdm.pandas()


morph = pymorphy2.MorphAnalyzer()
tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')


def remove_tags(text: str) -> str:
    return re.sub('<[^>]*>', ' ', text)


def review_to_wordlist(review, remove_stopwords=False):
    review = remove_tags(review)
    
    # убираем ссылки вне тегов
    review = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", review)
    # достаем сам текст
    
    review_text = re.sub("[^а-яА-Яa-zA-Z]"," ", review)
    # приводим к нижнему регистру и разбиваем на слова по символу пробела
    words = review_text.lower().split()
    
    if remove_stopwords:
      # убираем стоп-слова
        stops = stopwords.words("russian")
        words = [w for w in words if not w in stops]
    
    return words


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    if pd.isna(review):
        return []
    
    raw_sentences = tokenizer.tokenize(review.strip())
    
    sentences = []
    
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    
    return sentences


def clean_sentences(texts, verbose=True):
    return list(tqdm(map(
        lambda x: review_to_sentences(x, tokenizer), 
        texts
    ), desc='Чистка текста', total=len(texts), leave=False,))


def normalize_sentences(texts, except_words=None, divide_sentences=False, verbose=True):
    except_words = set(except_words or [])
    
    clean_sents = clean_sentences(texts, verbose=verbose)
    
    unique_words = set()
    for texts in clean_sents:
        [unique_words.update(text) for text in texts]
    
    unique_words.difference_update(except_words)
    words_norm = {
        word: morph.parse(word)[0].normal_form 
        for word in tqdm(unique_words, desc='Нормализация', leave=False)
    }
    
    normal_sents = []

    for texts in clean_sents:
        normal_sents.append([[words_norm.get(word, word) for word in text] for text in texts])
        
        if not divide_sentences:
            normal_sents[-1] = " ".join([" ".join([word for word in text]) for text in normal_sents[-1]])
            
    return normal_sents