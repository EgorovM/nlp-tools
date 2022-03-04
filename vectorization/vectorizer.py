from typing import List, Callable

import gensim
import numpy as np

from tqdm.notebook import tqdm

ISC_MODEL_PATH = 'models/isc_word2vec.pickle'


class ISCVectorizer:
    def __init__(self, model_path: str = None) -> None:
        self.model_path = model_path or ISC_MODEL_PATH
        self.model = gensim.models.Word2Vec.load(self.model_path)
        self._preprocessor = lambda text: text
        self._batch_preprocessor = lambda texts: texts
    
    def set_preprocessor(self, func: Callable) -> None:
        self._preprocessor = func
    
    def set_batch_preprocessor(self, func: Callable) -> None:
        self._batch_preprocessor = func
        
    def _preprocess(self, text: str) -> str:
        return self._preprocessor(text)
    
    def text_to_vec(self, text: str, preprocess: bool = True) -> np.ndarray:
        if preprocess:
            text = self._preprocess(text)
        
        vector = np.array([.0 for _ in range(self.model.vector_size)])
        count = 0
        
        for word in text.split():
            if word in self.model.wv:
                vector += self.model.wv[word]
                count += 1
        
        if count != 0:
            vector /= count
            
        return vector

    def texts_to_vec(self, texts: List[str]) -> List[np.ndarray]:
        texts = self._batch_preprocessor(texts)
        return list(tqdm(
            map(lambda text: self.text_to_vec(text, preprocess=False), texts),
            total=len(texts)
        ))
