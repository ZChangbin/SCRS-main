import os
import logging

import numpy as np
import wandb
import torch
import torch.nn.functional as F
from scipy.special import log_softmax

from rouge import Rouge
from nltk import ngrams
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


import warnings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
warnings.filterwarnings("ignore", category=FutureWarning)  
warnings.filterwarnings("ignore", category=UserWarning)   
from torch_scatter import scatter_add


class BaseEntailment:
    def save_prediction_cache(self):
        pass


from itertools import combinations
from scipy.sparse.csgraph import connected_components
class Deberta4SE():
    def __init__(self, tokenizer, model, device = None):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()  

    def get_relations(self,generated_sequences): 
        
        relations = []  
        for responses in generated_sequences:
            
            row, col = np.triu_indices(len(responses), k=1)
            forward_pairs = list(zip(responses[row], responses[col]))  
            backward_pairs = list(zip(responses[col], responses[row]))  
            text_pairs = forward_pairs + backward_pairs  
                
            
            inputs = self.tokenizer(text_pairs, return_tensors="pt", padding=True, truncation=True, max_length=80).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            
            softmax_logits = F.softmax(logits, dim=1)
            predictions = torch.argmax(softmax_logits, dim=1).cpu().numpy()  
            
           
            mid = len(predictions) // 2  
            first_half = predictions[:mid] 
            second_half = predictions[mid:]  
            
            
            result = ((first_half * second_half) > 1).astype(int)
            relations.append(result.tolist())
        
        return np.array(relations)


    
    def get_se_id(self,relations, num_generation):
        pairs = np.array(list(combinations(range(num_generation), 2)))
    
        num_relations = relations.shape[0]  
        num_pairs = relations.shape[1]  
    
        
        adj_matrices = np.zeros((num_relations, num_generation, num_generation))
    
        
        relation_indices = np.where(relations == 1)
        rows, cols = pairs[relation_indices[1]].T
    
        adj_matrices[relation_indices[0], rows, cols] = 1
        adj_matrices[relation_indices[0], cols, rows] = 1  
    
        def compute_connected_components(adj_matrix):
            n_components, labels = connected_components(adj_matrix, directed=False, return_labels=True)
            return labels
    
        labels = np.array(list(map(compute_connected_components, adj_matrices)))
        return labels


    
    def get_semantic_entropy(self, semantic_ids, log_likelihoods, device="cuda"):
        semantic_ids = torch.tensor(semantic_ids, device = device)
        log_likelihoods = torch.tensor(log_likelihoods, device = device)
        log_like_norm = torch.log_softmax(log_likelihoods, dim=1).to(torch.float32)  
        num_samples, num_groups = semantic_ids.shape
    
        i_flat = torch.repeat_interleave(torch.arange(num_samples, device=device), num_groups) 
        semantic_ids_flat = semantic_ids.view(-1) 
        log_like_norm_flat = log_like_norm.view(-1) 
    
        unique_pairs, inverse_indices = torch.unique(torch.stack((i_flat, semantic_ids_flat), dim=1), dim=0, return_inverse=True)
        logsumexp_values = torch.zeros(len(unique_pairs), dtype=torch.float64, device=device)
      
        logsumexp_values.index_add_(0, inverse_indices, torch.exp(log_like_norm_flat).to(torch.float64))  
  
        logsumexp_values = torch.log(logsumexp_values)

        probs = torch.exp(logsumexp_values)
  
        entropy_contribs = -probs * logsumexp_values    

        entropys = torch.zeros(num_samples, dtype=torch.float64, device=device)
        entropys.index_add_(0, unique_pairs[:, 0], entropy_contribs)  
    
        return entropys

    def get_naive_entropy(self, log_likelihoods):
        log_like_norm = log_likelihoods - torch.logsumexp(log_likelihoods, dim=1, keepdim=True)
        probs = torch.exp(log_like_norm)
        entropys = -(probs * log_like_norm)
        return entropys

    def compute_single_response_entropy(log_likelihood_single: torch.Tensor) -> torch.Tensor:
        log_probs = log_likelihood_single - torch.logsumexp(log_likelihood_single, dim=0, keepdim=True)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum()
        return entropy


def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy

