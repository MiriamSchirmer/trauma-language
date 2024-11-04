### Implementation of the standard classification models.
import typing as tp
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from datasets import DatasetDict
from abc import ABC, abstractmethod
from multiprocessing import Pool
from transformers import AutoTokenizer
from code.transformer_models import Bert, RoBERTa
from code.dataset import load_gtc
from code.transformer_models import Trainer
from code.bow_model import NaiveBayesEstim, BoW

class ClassificationModel(ABC):
    """ Define the interface for the classification models. (scikit-learn style)"""

    @abstractmethod
    def load(file_name: str):
        """ Load a classification model from a file. 
            It should be possilbe to initialize the model with default parameters,
            i.e., c = ClassificationModel()
            and then perform c.load(filename) to load any model of this class.
        """
        pass

    def predict(self, input_sequences: tp.List[str]) -> np.ndarray:
        """ 
            input_sequences: array of length N
            returns np array of length [N] with the class number
        """
        return np.argmax(self.predict_proba(input_sequences), axis=1)

    @abstractmethod
    def predict_proba(input_sequences: tp.List[str]) -> np.ndarray:
        """ 
            input_sequences: array of length N
            returns np array of length [N, K] with the class probabilities for K classes
        """
        pass

    @abstractmethod
    def train(self, ds: DatasetDict, **kwargs):
        """ Train the predictive model.
            ds: DatasetDict containing {"train", "test"}. The train method
            should make sure that only the "train" part is used for the actual training.
        """
        pass
    
class FeedForwardModel(ClassificationModel):
    def __init__(self, tokenizer=None, hidden_dim1=100, hidden_dim2 = None, device="cuda:3"):
        self.bow = BoW(tokenizer)
        self.hidden_dim1=hidden_dim1
        if hidden_dim2 == "none":
            hidden_dim2 = None
        self.hidden_dim2=hidden_dim2
        self.use_device = device
        self.input_dim = None # yet unknown

    def train(self, ds: DatasetDict, num_eps=10, lr=5e-4, batch_size=32):
        return self.fit_sgd(ds, num_eps, lr, batch_size)

    def fit_sgd(self, ds: DatasetDict, num_eps=10, lr=5e-4, batch_size=32):
        X_train_bow, X_train, y_train, X_test_bow, X_test, y_test = self.bow.convert_ds_to_tfidf(ds)
        train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train_bow.toarray()).float(), torch.tensor(y_train).long())
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
        test_ds = torch.utils.data.TensorDataset(torch.tensor(X_test_bow.toarray()).float(), torch.tensor(y_test).long())
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

        self.input_dim = X_train_bow.shape[1]
        #print(self.input_dim)
        self.offset = np.mean(X_train_bow.toarray())
        self.ffnn = TwoLayerTFIDF(self.input_dim, hidden_dim1=self.hidden_dim1, hidden_dim2=self.hidden_dim2, offset=self.offset)
        self.ffnn = self.ffnn.to(self.use_device)
        my_optim = Adam(self.ffnn.parameters(), lr = lr)
        losses = []
        iters = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        for ep in range(num_eps):
            losses = []
            for batch in train_loader:
                my_optim.zero_grad()
                #output = example_model.forward(batch["input_ids"])
                #loss = myloss(output, batch["label"])
                #print(input_embs.shape)
                mydata, mylabels = batch
                output = self.ffnn.forward(mydata.to(self.use_device))
                loss = loss_fn(output, mylabels.to(self.use_device))
                loss.backward()
                my_optim.step()
                losses.append(loss.item())
                iters += 1
            print(torch.tensor([losses]).mean())
        print("Test accuracy: ", self._test(test_loader))

    def _test(self, my_dl):
        model = self.ffnn.to(self.use_device)
        #print(len(my_dl))
        corr, total = 0, 0
        for mydata, mylabels in my_dl:
            #print(mydata[0])
            output = model.forward(mydata.to(self.use_device))
            corr += torch.sum(output.argmax(dim=-1) == mylabels.to(self.use_device)).item()
            total += len(mydata)
            #print(output.argmax(dim=-1), mylabels)
        return corr/total

    def predict_proba(self, input_sequences: tp.List[str]):
        self.ffnn = self.ffnn.to(self.use_device)
        X_test_bow = self.bow.tfidf_vec.transform(input_sequences)
        test_ds = torch.utils.data.TensorDataset(torch.tensor(X_test_bow.toarray()).float())
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)
        res_list = []
        with torch.no_grad():
            for mydata in test_loader:
                #print(mydata[0])
                output = self.ffnn.forward(mydata[0].to(self.use_device))
                res_list.append(output.detach())
            res_list = torch.cat(res_list, dim=0)
        return torch.softmax(res_list, axis=-1).cpu().numpy()

    def save(self, filename):
        save_dict = {
            "hidden_dim1": self.hidden_dim1,
            "hidden_dim2": self.hidden_dim2 if type(self.hidden_dim2) == int else "none",
            "input_dim": self.input_dim,
            "tfvec": self.bow.tfidf_vec,
            "offset": self.ffnn.offset
        }
        save_dict["state_dict"] = self.ffnn.state_dict()
        torch.save(save_dict, filename)
    
    def load(self, filename):
        params_dict = torch.load(filename, map_location=self.use_device)
        self.bow = BoW(None)
        self.bow.tfidf_vec = params_dict["tfvec"]
        self.hidden_dim1 = params_dict["hidden_dim1"]
        self.hidden_dim2 = params_dict["hidden_dim2"]
        if self.hidden_dim2 == "none":
            self.hidden_dim2 = None
        self.input_dim = params_dict["input_dim"]
        offset = params_dict["offset"]
        self.ffnn = TwoLayerTFIDF(self.input_dim, hidden_dim1=self.hidden_dim1, hidden_dim2=self.hidden_dim2, offset=offset)
        self.ffnn.load_state_dict(params_dict["state_dict"])


### Feed-forward model
class TwoLayerTFIDF(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2 = None, offset=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim1))
        if hidden_dim2 is None:
            self.layers.append(nn.Linear(hidden_dim1, 2))
        else:
            self.layers.append(nn.Linear(hidden_dim1, hidden_dim2))
            self.layers.append(nn.Linear(hidden_dim2, 2))
        self.offset = offset

    def forward(self, x):
        for i, lay in enumerate(self.layers):
            x = lay(x)
            if i < len(self.layers) - 1: ## all but last
                x = torch.relu(x)
        return x - self.offset


### BERT model

class BertBased(ClassificationModel):
    def __init__(self, n_layers=6, n_heads=12, d_hidden=768, max_seq_len=512, device="cuda:2"):
        self.device = device
        self.max_seq_len = max_seq_len

    def train(self, ds, lr=1e-5, epochs = 3, batch_size=8):
        trainer = Trainer(ds=ds, model=self.model, tokenizer=self.tokenizer, device=self.device, max_seq_len=self.max_seq_len, tokenization_required=True)
        step_cnt = 0
        for ep in range(epochs):
            step_cnt = trainer.train(epochs=1, test_interval=1, batch_size=batch_size, lr=lr, start_steps = step_cnt, start_epoch=ep)
        self.model.model = trainer.get_model()

    def predict_proba(self, input_sequences):
        self.model.model = self.model.model.to(self.device)
        input_sequences =  list([str(seq) for seq in input_sequences])
        int_batch = 32
        my_res_list = []
        i = 0
        with torch.no_grad():
            while i*int_batch < len(input_sequences):
                inputs = self.tokenizer(input_sequences[i*int_batch:(i+1)*int_batch], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_seq_len,)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                #print(inputs.keys())
                outputs = self.model.model(**inputs)["logits"]
                my_res_list.append(torch.softmax(outputs, dim=-1).cpu().numpy())
                i+=1
        return np.concatenate(my_res_list, axis=0)

    def save(self, filename):
        save_dict = {
            "n_layers": self.model.config.num_hidden_layers,
            "n_heads": self.model.config.num_attention_heads,
            "d_hidden": self.model.config.hidden_size,
            "max_seq_len": self.max_seq_len
        }
        save_dict["state_dict"] = self.model.model.state_dict()
        torch.save(save_dict, filename)
    
    def load(self, filename):
        params_dict = torch.load(filename, map_location=self.device)
        self.model = Bert(n_layers=params_dict["n_layers"], n_heads=params_dict["n_heads"], dim=params_dict["d_hidden"])
        self.model.model.load_state_dict(params_dict["state_dict"])



class RoBERTamodel(BertBased):
    def __init__(self, n_layers=6, n_heads=12, d_hidden=768, max_seq_len=512-2, device="cuda:2"):
        super().__init__(n_layers, n_heads, d_hidden, max_seq_len, device)
        self.tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base', use_fast=True, padding='max_length',
                truncation=True, max_length=self.max_seq_len, return_tensors='pt')

        self.model = RoBERTa(n_layers=n_layers, n_heads=n_heads, dim=d_hidden)
        self.model.model = self.model.model.to(device)

    def load(self, filename):
        params_dict = torch.load(filename, map_location=self.device)
        self.model = RoBERTa(n_layers=params_dict["n_layers"], n_heads=params_dict["n_heads"], dim=params_dict["d_hidden"])
        self.model.model.load_state_dict(params_dict["state_dict"])

    def train(self, ds, lr=1e-5, epochs = 5, batch_size=8):## Default 5 epochs
        super().train(ds, lr, epochs, batch_size)

class RoBERTaPretrainedmodel(RoBERTamodel):
    def __init__(self, n_layers=6, device="cuda:2"):
        super().__init__(n_layers, device=device)
        self.model = RoBERTa(n_layers=n_layers, pretrained=True)

    def load(self, filename):
        params_dict = torch.load(filename, map_location=self.device)
        self.model = RoBERTa(n_layers=params_dict["n_layers"], pretrained=True)
        self.model.model.load_state_dict(params_dict["state_dict"])
    
    def train(self, ds, lr=1e-5, epochs = 3, batch_size=8):## Default 5 epochs
        super().train(ds, lr, epochs, batch_size)

class BERTmodel(BertBased):
    def __init__(self, n_layers=6, n_heads=12, d_hidden=768, max_seq_len=512, device="cuda:2"):
        super().__init__(n_layers, n_heads, d_hidden, max_seq_len, device)
        self.model = Bert(n_layers=n_layers, n_heads=n_heads, dim=d_hidden)
        self.model.model = self.model.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, padding='max_length',
                truncation=True, max_length=self.max_seq_len, return_tensors='pt')

    def train(self, ds, lr=1e-5, epochs = 5, batch_size=8): ## Default 5 epochs
        super().train(ds, lr, epochs, batch_size)

class BERTPretrainedmodel(BERTmodel):
    def __init__(self, n_layers=6, device="cuda:2"):
        super().__init__(n_layers, device=device)
        self.model = Bert(n_layers=n_layers, pretrained=True)

    def train(self, ds, lr=1e-5, epochs = 3, batch_size=8):## Default 5 epochs
        print("Training 3 epochs.")
        super().train(ds, lr, epochs, batch_size)

### Naive Bayes Model

class NaiveBayesBOWmodel(ClassificationModel):
    def __init__(self, tokenizer=None, multiplicities=True, device=None):
        self.multiplicities = multiplicities
        self.tokenizer = tokenizer
        self.myBoW = None
        self.mymodel = None

    def train(self, ds, alpha=2.0):
        self.myBoW = BoW(self.tokenizer)
        self.mymodel = NaiveBayesEstim(self.myBoW, alpha=alpha, multiplicities=self.multiplicities, imbalanced=True)
        self.mymodel.fit(ds)

    def load(self, filename):
        my_dict = torch.load(filename)
        self.myBoW = BoW(None)
        self.myBoW.count_vec = my_dict["count_vec"]
        self.mymodel = NaiveBayesEstim(self.myBoW, alpha=my_dict["alpha"], multiplicities=my_dict["mutiplicities"], imbalanced=True)
        self.mymodel.p1 = my_dict["p1"]
        self.mymodel.coef_ = my_dict["coef"]

    def save(self, filename):
        resdict = {"count_vec": self.myBoW.count_vec, "p1": self.mymodel.p1, "coef": self.mymodel.coef_, "mutiplicities": self.multiplicities, "alpha": self.mymodel.alpha}
        torch.save(resdict, filename)

    def predict_proba(self, input_sequences):
        return self.mymodel.predict_proba(input_sequences)



from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

### Logistic Regression Model
class NGramLogisticRegression(ClassificationModel):
    def __init__(self, tokenizer=None, n_gram_range=(1,1), device=None):
        self.mylr = None
        self.tokenizer = None
        self.n_gram_range = n_gram_range
        self.count_vec = CountVectorizer(ngram_range=tuple(n_gram_range))

    def train(self, ds, penalty="l2", C=1.0):
        if penalty == "none":
            penalty = None
        self.mylr = LogisticRegression(penalty=penalty, C=C, solver="lbfgs" if penalty!="l1" else "saga", max_iter=1000)
        X_train_vect = self.count_vec.fit_transform(ds["train"]["text"])
        y_train = np.array(ds["train"]["label"])
        print("Obtained feature vectors of shape", X_train_vect.shape)
        self.mylr.fit(X_train_vect, y_train)

    def save(self, filename):
        res_dict = {"count_vec": self.count_vec, "lr.coef_": self.mylr.coef_, "lr.intercept_": self.mylr.intercept_}
        torch.save(res_dict, filename)

    def load(self, filename):
        my_dict = torch.load(filename)
        self.count_vec = my_dict["count_vec"]
        self.mylr = LogisticRegression()
        self.mylr.coef_ = my_dict["lr.coef_"]
        self.mylr.intercept_ = my_dict["lr.intercept_"]
        self.mylr.classes_ = np.array([0,1]) # Shows sklearn that the model is trained.

    def predict_proba(self, input_sequences):
        X_test = self.count_vec.transform(input_sequences)
        return self.mylr.predict_proba(X_test)


### GPT-API model
from openai import OpenAI
import os
import json
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
API_LEGACY_ENDPOINT = "https://api.openai.com/v1/completions"

DEFAULT_PROMPT = "You are tasked with detecting trauma in text segments of transcripts of genocide tribunals. Specifically, detect instances that meet the APA’s definition of trauma. Psychological trauma, as defined by the APA, includes experiences of exposure to actual or threatened death, serious injury, or sexual violence, either directly encountered or witnessed. It also includes instances where individuals learn that the traumatic event(s) occurred to a close family member or friend. Label the text with '1' if there are indicators of trauma based on this definition, and '0' if there are no indicators of trauma. Note that trauma is rare and occurs in less than 20% of the cases. Only answer with either '0' or '1'."


class OpenAImodel(ClassificationModel):
    """ Use the OpenAI API to perform classifications. """

    def __init__(self, target_model="gpt4-turbo", prompt_template=DEFAULT_PROMPT, openai_keys_file="openai.json", n_threads=6, device=None):
        """
            target_model: the API to use
            prompt template: the prompt template to use. will be used with (template % sequence) to generate prompt. Should contain the command to just output either "Yes" or "No".
            openai_keys: tuple of (openai_org, openai_apikey)
        """
        self.n_threads = n_threads
        self.model = target_model
        self.prompt_template = prompt_template
        
        if openai_keys_file is not None:
            credentials = json.load(open(openai_keys_file))
            self.openai_org, self.openai_key = credentials["org"], credentials["key"]
            self.client = OpenAI(
                organization=self.openai_org,
                api_key =self.openai_key
            )
        else:
            elf.openai_org, self.openai_key = None, None

    def train(self, ds):
        pass # No training requred

    def predict_proba(self, input_sequences):
        #print(len(input_sequences))
        #with p as Pool(self.n_threads):
        #f_map = lambda seq: self._predict_proba_single(input_sequences, seq)
        tasklist =  list([(self.openai_org, self.openai_key, self.model, self.prompt_template, i) for i in input_sequences])
        with Pool(self.n_threads) as p:
            res_array = p.map(predict_proba_single, tasklist)
        return np.stack(res_array, axis=0)

    def predict(self, input_sequences):
        if self.client is None:
            raise ValueError("Client is not initialized. Please pass an API key to the constructor or load a model file.")
        message_dict = [{
            "role": "system",
            "content": self.prompt_template
        },
        {
            "role": "user",
            "content": None
        }]
        res_array = []
        for seq in input_sequences:
            message_dict[1]["content"] = seq #  self.prompt_template.format(seq)
            results = self.client.chat.completions.create(
                model=self.model,
                messages=message_dict,
                logprobs=True,
                temperature=0.000000001
            )
            print(results.choices[0])
            res_array.append((results.choices[0].message.content, results.choices[0].logprobs))
        return res_array

        
    def save(self, filename):
        torch.save({"model": self.model, "prompt": self.prompt_template, "api_key": self.openai_key, "api_org": self.openai_org}, filename)

    def load(self, filename):
        res = torch.load(filename)
        self.model=res["model"]
        self.prompt_template =res["prompt"]
        self.openai_key = res["api_key"]
        self.openai_org = res["api_org"]
        self.client = OpenAI(
                organization=self.openai_org,
                api_key = self.openai_key
            )

    
""" Subthread. """

def compute_probas_from_logprobs(choice_logprobs):
    """ Compute the probabilities of chosing the labels 0, 1. """
    p1 = -float("inf")
    p0 = -float("inf")
    for token in choice_logprobs:
        if token.token == "1":
            p1 = token.logprob
        if token.token == "0":
            p0 = token.logprob
    #print("0", p0, "1", p1)
    odds_diff = p1-p0
    if np.isnan(odds_diff):
        p1 = 0.5
    elif odds_diff == float("inf"):
        p1 = 1.0
    elif odds_diff == -float("inf"):
        p1 = 0.0
    else:
        p1 = np.exp(odds_diff)/(np.exp(odds_diff)+1)
    return np.array([1-p1, p1])


def predict_proba_single(arglist):
    openai_org, openai_key, target_model, prompt_template, predict_seq = arglist
    client = OpenAI(
                organization=openai_org,
                api_key =openai_key
            )
    if client is None:
        raise ValueError("Client is not initialized. Please pass an API key to the constructor or load a model file.")

    message_dict = [{
        "role": "system",
        "content": prompt_template
    },
    {
        "role": "user",
        "content": None
    }]

    message_dict[1]["content"] = predict_seq #self.prompt_template.format(seq)
    #print(message_dict)
    results = client.chat.completions.create(
        model=target_model,
        messages=message_dict,
        logprobs=True,
        top_logprobs = 5,
        temperature=0.00001,
        seed=1
    )
    return compute_probas_from_logprobs(results.choices[0].logprobs.content[0].top_logprobs)
    #print(results.choices[0])
