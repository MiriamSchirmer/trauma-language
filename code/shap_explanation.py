from nltk.tokenize import word_tokenize
import numpy as np
import torch
from typing import List, Tuple, Dict, Union
import regex
import shap

class ShapleyValues():
    """ A class to compute ground truth Shapley values. """
    def __init__(self, model, tokenizer_func=word_tokenize, num_samples: Union[int, str] = "auto", method="partition", impute_token="[UNK]", device="cuda"):
        """
            We implement two sampling strategies for Shapley values,
            Partition (method="partition") Permutation (method="permutation")
        """
        self.model = model
        self.tokenizer_func = tokenizer_func
        self.num_samples = num_samples
        self.impute_token = impute_token
        self.method = method
        #self.bg = np.array([bg]) # background data used for kernel shap
        self.device = device

    def get_signed_importances(self, input_sample: Union[str, list]):
        res = self.get_shap_values(input_sample)
        res_dict = {}
        for i in range(len(res.values)):
            res_dict[res.data[i]] = res.values[i]
             #.values[0]
        return res_dict

    def get_shap_values(self, input_sample: Union[str, list]):
        def f_batch(x): # x is a list of strings
            if type(x)==str:
                x = [x]
            probas = self.model.predict_proba(x)
            #probas = outputs.detach().cpu().numpy()
            eps=0.001
            val = np.log((probas[:,1]+eps)/(probas[:,0]+eps))
            return val


        my_masker = shap.maskers.Text(tokenizer=None, mask_token=self.impute_token)
        test = {'text': [input_sample]}

        if self.method=="permutation":
            explainer_bert = shap.PermutationExplainer(f_batch, masker=my_masker)

        elif self.method=="partition":
            explainer_bert = shap.PartitionExplainer(f_batch, masker=my_masker)

        shap_values = explainer_bert(test, max_evals = self.num_samples)
        return shap_values[0] #.values[0]