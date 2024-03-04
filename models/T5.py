import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer

pl.seed_everything(42)
#  generator = T5Gen(tokenizer_path, model_path, max_length, device, False)

class T5_Model(pl.LightningModule):
    def __init__(self, tokenizer_path, model_path, max_length, sep_token):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.max_length = max_length
        self.sep_token = sep_token

    def forward(self, source, targets=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("********\n")
        # print(device)
        # print("********\n")
        # prepare
        for i, src in enumerate(source):
            prepared_source = ' '.join([self.sep_token, src])
            source[i] = prepared_source
        # tokenize
        model_inputs = self.tokenizer(source, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(device)
        # Predict
        self.model = self.model.to(device) 
        if targets:
            labels = self.tokenizer(targets, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").input_ids.to(device)
            labels[labels == 0] = -100  # Useful for T5 to pad the 0 labels
            # Predict
            output = self.model(**model_inputs, labels=labels) # forward pass
            # output = self.model(input_ids=model_inputs['input_ids'], attention_mask=model_inputs["attention_mask"], labels=labels) # forward pass
        else:
            generated_ids = self.model.generate(**model_inputs, 
                                                max_length=self.max_length, 
                                                num_beams=2,
                                                length_penalty=1.0,
                                                early_stopping=True)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return output