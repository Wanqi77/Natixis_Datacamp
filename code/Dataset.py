import copy
from unicodedata import name
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from transformers import DistilBertTokenizer
import transformers

class PredictDataset(Dataset):

    def __init__(self, data_file, test_size, max_len, tokenizer):
        self.max_len = max_len # use to padding all sequence to a fixed length
        self.tokenizer = tokenizer
        with open(data_file, 'r') as fp:
            data = json.load(fp)
        
        self.data = []
        for data_dict in data:
            s = data_dict['speech'][-1]
            if s['ECB']:
                l = s['ECB'][0].strip()
            else:
                l = s['FED'][0].strip()
            #words = l.split(' ')
            text_token = self.tokenizer(l, padding='max_length', truncation=True, max_length=self.max_len)
            #print(text_token)

            text = torch.tensor(text_token['input_ids'], dtype=float)
            text_att = torch.tensor(text_token['attention_mask'], dtype=int)
            stock = torch.tensor(data_dict['stock'], dtype=float)
            tgt_c = torch.tensor(data_dict['target_classif'], dtype=int)
            tgt_r = torch.tensor(data_dict['target_reg'], dtype=float)

            
            self.data.append({'text_token':text, 'text_att':text_att, 'stock':stock, 'target_classif':tgt_c, 'target_reg':tgt_r})
    

        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size,random_state=0)
        print("="*50)
        print("Data Preprocess Done!")
        print("Dataset size:{}, train:{}, val:{}".
              format(len(self.data),len(self.train_data),len(self.test_data)))
        print("="*50)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def train_set(self):
        '''call this method to switch to train mode'''
        self.data = self.train_data
        return copy.deepcopy(self)

    def test_set(self):
        '''call this method to switch to test mode'''
        self.data = self.test_data
        return copy.deepcopy(self)



def main():
    model = "distilbert-base-uncased"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    dataset = PredictDataset('sums.json', 0.2, 512, tokenizer)

if __name__ == '__main__':
    main()

