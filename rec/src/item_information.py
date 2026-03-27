import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torch.utils.data import DataLoader, TensorDataset
from transformers import  AutoTokenizer, AutoModel

class item_comment():
    def __init__(self, dataset, model_name_or_path=None):
        if model_name_or_path is None:
            model_name_or_path = 'roberta-base'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(DEVICE)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if dataset == 'redial':
            json_path = os.path.join(base_dir, 'process_item_comments_redial_sen.json')
            with open(json_path, "r", encoding="utf-8") as f:
                self.process_item_comments = json.load(f)
        elif dataset == 'inspired':
            json_path = os.path.join(base_dir, 'process_item_comments_inspired_sen.json')
            with open(json_path, "r", encoding="utf-8") as f:
                self.process_item_comments = json.load(f)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.sentence = [comments for id_, comments in self.process_item_comments.items()]
    def get_item_embedding(self, max_length=200, batch_size=8):

        inputs = self.tokenizer(self.sentence, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        all_embeddings = []
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :] 
                all_embeddings.append(cls_embedding)


        all_embeddings = torch.cat(all_embeddings, dim=0)

        del self.model
        torch.cuda.empty_cache()
        
        return all_embeddings


class SemanticMapping(nn.Module):
    def __init__(self, hidden_size, device):
        super(SemanticMapping, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.entity_proj_shared = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        self.token_proj_shared = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)


        self.entity_proj1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
        )
        self.token_proj1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
        )
   
        self.cross_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)

    def forward(self, entity_embeds=None, token_embeds=None):

        if entity_embeds is not None:
            entity_embeds = entity_embeds.to(self.device)
            entity_embeds = self.entity_proj_shared(entity_embeds).to(self.device)  
        if token_embeds is not None:
            token_embeds = token_embeds.to(self.device)
            token_embeds = self.token_proj_shared(token_embeds).to(self.device) 
        

        attn_weights = self.cross_attn(token_embeds) @ entity_embeds.T 

        attn_weights /= self.hidden_size 

        item_embeds = entity_weights @ entity_embeds + token_embeds 
    
        return item_embeds

if __name__ == '__main__':
    item_comment = item_comment(dataset = 'inspired')
    item_embedding = item_comment.get_item_embedding()
    print(item_embedding.shape)
    