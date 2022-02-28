import os
import re

from typing import List, Dict, Union, Callable, Iterable

import pandas as pd

import nltk.data
from nltk.corpus import stopwords as nltk_stopwords

import joblib
import pymorphy2
from tqdm.notebook import tqdm


class CachedNormalizer:
    def __init__(
            self,
            tokenizer=None,
            word2norm: Dict[str, str] = None,
            remove_stopwords: bool = False,
            stopwords: List[str] = None,
    ) -> None:
        self.tokenizer = tokenizer or nltk.data.load('tokenizers/punkt/russian.pickle')
        self.word2norm = word2norm or dict()
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords or nltk_stopwords.words("russian")

        self.morph = pymorphy2.MorphAnalyzer()

    def set_word_lemmatizing(self, lemmatize_func: Callable) -> None:
        self.lemmatize_word = lemmatize_func

    def lemmatize_word(self, word: str) -> str:
        """ Приводит слово к его начальной форме. Сохраняет уже просмотренные слова. """
        if word not in self.word2norm:
            self.word2norm[word] = self.morph.parse(word)[0].normal_form

        return self.word2norm[word]

    def text_to_wordlist(self, text: str) -> List[str]:
        # уберем теги
        text = re.sub('<[^>]*>', ' ', text)

        # убираем ссылки вне тегов
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            " ",
            text
        )

        # достаем сам текст
        text = re.sub("[^а-яА-Яa-zA-Z0-9/.]", " ", text)

        # приводим к нижнему регистру и разбиваем на слова по символу пробела
        words = text.lower().split()

        if self.remove_stopwords:
            # убираем стоп-слова
            words = [w for w in words if w not in self.stopwords]

        return words

    def text_to_sentences(self, text: str) -> List[List[str]]:
        if pd.isna(text):
            return []

        raw_sentences = self.tokenizer.tokenize(text.strip())

        sentences = []

        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                words = self.text_to_wordlist(raw_sentence)
                sentences.append(words)

        return sentences

    def clean_sentences(self, texts: List[str]) -> List[List[str]]:
        return list(tqdm(map(
            lambda x: self.text_to_sentences(x),
            texts
        ), desc='Чистка текста', total=len(texts), leave=False))

    def normalize(
        self, 
        texts: List[str], 
        as_sentences: bool = False, 
        except_words: Iterable[str] = None
    ) -> Union[List[str], List[List[str]]]:
        """ Лемматизация текстов
        :param texts: тексты, которые нужно отнармализовать
        :param as_sentences: нужно ли разделить по предложениям, если True, то да
        :param except_words: слова, которые не нужно нормализовать (например аббревиатуры)
        """
        except_words = set(except_words or [])
        clean_sents = self.clean_sentences(texts)

        unique_words = set()
        for texts in clean_sents:
            [unique_words.update(text) for text in texts]
        
        for word in tqdm(unique_words, desc='Лемматизация слов', leave=False):
            self.lemmatize_word(word)

        normal_sents = []

        transform = lambda x: x
        if not as_sentences:
            transform = lambda x: " ".join(x)

        for texts in clean_sents:
            text = transform([
                transform([
                    word
                    if word in except_words

                    else
                    self.word2norm.get(word, word)

                    for word in text
                ])
                for text in texts
            ])

            normal_sents.append(text)

        return normal_sents
    
    def save(self, dir_path):
        joblib.dump(self.word2norm, os.path.join(dir_path, 'word2norm.joblib'))
        
    @staticmethod
    def load(dir_path):
        word2norm = joblib.load(os.path.join(dir_path, 'word2norm.joblib'))
        
        return CachedNormalizer(word2norm=word2norm)