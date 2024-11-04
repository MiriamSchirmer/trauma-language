from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, BertTokenizerFast
import datasets
import numpy as np
import torch
from typing import List, Tuple, Dict, Union
import regex
import shap
import torch
import torch.nn.functional as F
import math
import typing as tp


class BoW():
    """ Instantiate a Bow Representation of the dataset, which can subsequently be used to perform different tasks
        e.g., build a NaiveBayes or Feedforward Model of the data.
    """
    def __init__(self, tokenizer: tp.Union[AutoTokenizer, None, tp.Callable] = None,
            min_df: int=1, token_pattern: str=r'[\S]+' , max_seq_len: int=512) -> None: 
        """
            tokenkizer: None: word level tokenizer OR huggingface transformers AutoTokenizer OR str -> list function.
        """
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        print(f'self.tokenizer: {self.tokenizer} type(self.tokenizer: {type(self.tokenizer)}')
        self.count_vec = CountVectorizer(tokenizer=self._custom_tokenizer, token_pattern=None)
        self.tfidf_vec = TfidfVectorizer(min_df=min_df, tokenizer=self._custom_tokenizer, token_pattern=None)
        
        #self.X_train_bow, self.X_train, self.y_train, self.X_test_bow, self.X_test, self.y_test = self.__convert_ds_to_tfidf(self.ds)

    def convert_tf_idf(input_seq: List[str]):
        return self.tfidf_vec.transform(input_seq)

    def convert_counts(input_seq: List[str]):
        return self.count_vec.transform(input_seq)

    #@profile
    def _custom_tokenizer(self, text: str) -> List[str]:
        """Custom tokenizer for TfidfVectorizer used for AutoTokenizer from HuggingFace"""
        if self.tokenizer is None:
            tokens = text.split()
            return tokens
        elif isinstance(self.tokenizer, BertTokenizerFast):
            #print("Autotok")
            tok_list = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
            special_list =  self.tokenizer.all_special_tokens
            return list(filter(lambda a: a not in special_list, tok_list))
        else:
            return self.tokenizer(text)
        # return regex.findall(r'\b\w+\b|.', text)
    
    def __ds_to_lists(self, ds: datasets.arrow_dataset.Dataset) -> Tuple[List[str], List[int]]:
        """Convert a HuggingFace dataset to a list of texts and labels"""
        texts = []
        labels = []

        for i in range(len(ds)):
            labels.append(int(ds[i]['label']))
            texts.append(ds[i]['text'])
        labels = [int(i) for i in labels]
        return texts, labels

    #@profile
    def convert_ds_to_tfidf(self, ds: datasets.dataset_dict.DatasetDict) -> Tuple[np.ndarray, List[str], List[int], np.ndarray, List[str], List[int]]:
        """Convert a HuggingFace dataset to a BoW representation using TfidfVectorizer
        
        Returns:
            X_train_bow: BoW representation of the training set
            X_train: List of texts of the training set
            y_train: List of labels of the training set
            X_test_bow: BoW representation of the test set
            X_test: List of texts of the test set
            y_test: List of labels of the test set
        """
        X_train, y_train = self.__ds_to_lists(ds['train'])
        X_test, y_test = self.__ds_to_lists(ds['test'])
        X_train_bow = self.tfidf_vec.fit_transform(X_train)
        X_test_bow = self.tfidf_vec.transform(X_test)
        return X_train_bow, X_train, y_train, X_test_bow, X_test, y_test

    #@profile
    def convert_ds_to_class_count(self, ds: datasets.dataset_dict.DatasetDict) -> Tuple[float, np.ndarray, np.ndarray]:
        """ Return class wise counts for the Naive bayes model:
            returns [p(y=1), token_counts in class 1, token_counts in class 0]
        """
        X_train, y_train = self.__ds_to_lists(ds['train'])
        counts = self.count_vec.fit_transform(X_train)
        #counts = self.count_vec.fit_transform(X_train)
        counts = (counts > 0).astype(int)
        occ_counts_pos = counts[y_train == 1].sum(axis=0)
        occ_counts_neg = counts[y_train == 0].sum(axis=0)
        return y_train.float().mean(), occ_counts_pos, occ_counts_neg
    
    def convert_ds_to_count_vectors(self, ds: datasets.dataset_dict.DatasetDict, norm_len=True):
        """ Compute overall counts of terms in documents. Then fit logistic model. """
        X_train, y_train = self.__ds_to_lists(ds['train'])
        #Xtok = self.__apply_autotokenizer(X_train)
        counts = self.count_vec.fit_transform(X_train)
        #print(counts.shape)
        if norm_len:
            total_doc_counts=counts.sum(axis=1) # total number of words in documents
            counts /= total_doc_counts
        return counts

class NaiveBayesEstim():
    def __init__(self, bow: BoW, alpha: float=20.0, multiplicities=False, imbalanced=False):
        self.bow = bow
        self.p1 = None # self.y_train.float().mean().item()
        #print(self.p1, self.bow)
        self.classes_ = [0,1]
        self.alpha = alpha
        self.coef_ = None
        self.multiplicities = multiplicities
        self.imbalanced = imbalanced
        self._estimator_type = "classifier" # pretend to be sklearn classifier
        
    def fit(self, ds: datasets.DatasetDict):
        counts = self.bow.convert_ds_to_count_vectors(ds, norm_len=False)
        y_train = torch.tensor(ds["train"]["label"])
        y_test = ds["test"]["label"]
        self.p1 = y_train.float().mean().item()
        #print(counts.shape)
        if self.multiplicities:
            counts = counts.toarray()
            poscnt = counts[y_train == 1].sum(axis=0) # total occurance of word in positive class
            negcnt = counts[y_train == 0].sum(axis=0) # total occurance of word in positive class
            pos_nocc = np.sum(counts[y_train == 1]) # total number of words in positive class
            neg_nocc = np.sum(counts[y_train == 0]) # total number of words in negative class
            # Share words 
            if self.imbalanced:
                offset = np.log(pos_nocc/neg_nocc)
            else:
                offset = 0.0
            #print(offset)
            self.coef_ = np.array(np.log((poscnt + self.alpha)/(negcnt + self.alpha))).flatten() - offset
        else:
            counts = (counts > 0).astype(int)
            poscnt = counts[y_train == 1].sum(axis=0) # docs with word in postive class 
            negcnt = counts[y_train == 0].sum(axis=0) # docs with word in negative class
            pos_docs = torch.sum(y_train == 1).item()
            neg_docs = torch.sum(y_train == 0).item()
            if self.imbalanced:
                offset = np.log(self.p1/(1-self.p1))
            else:
                offset = 0.0
            #print(offset)
            self.coef_ = np.array(np.log((poscnt + self.alpha)/(negcnt + self.alpha))).flatten() - offset
        #print(self.coef_)

    def sigmoid(self, X):
        return 1.0/(np.exp(-X)+1)
    
    def predict_proba(self, text_test):
        X = self.bow.count_vec.transform(text_test).toarray()
        if self.multiplicities == False:
            X = (X > 0).astype(int)
        X = X.astype(float)
        #print(X.dtype, self.coef_.shape, math.log(self.p1/(1.0-self.p1)))
        res = np.matmul(X, self.coef_.reshape(-1,1)) + math.log(self.p1/(1.0-self.p1))
        return self.sigmoid(res.reshape(-1,1)*np.array([-1,1]))
    
    def predict(self, text_test):
        if self.coef_ is None:
            self.fit()
        return np.argmax(self.predict_proba(text_test), axis=1)
        
    def get_importance(self) -> Dict[str, float]:
        #self.fit()
        top_indices_pos = np.argsort(np.abs(self.coef_))[::-1]
        feature_names = np.array(self.bow.count_vec.get_feature_names_out())
        return {tuple(feature_names[idx]) if type(feature_names[idx]) is not str else feature_names[idx]: np.abs(self.coef_[idx]) for idx in top_indices_pos}
    
    def get_signed_importance(self) -> Dict[str, float]:
        #self.fit()
        top_indices_pos = np.argsort(self.coef_)[::-1]
        feature_names = np.array(self.bow.count_vec.get_feature_names_out())
        return {tuple(feature_names[idx]) if type(feature_names[idx]) is not str else feature_names[idx]: self.coef_[idx] for idx in top_indices_pos}