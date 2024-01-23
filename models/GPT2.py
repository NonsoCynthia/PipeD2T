import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import GPT2LMHeadModel, GPT2TokenizerFast as GPT2Tokenizer

pl.seed_everything(42)

class GPT2_Model(pl.LightningModule):
    def __init__(self, tokenizer_path, model_path, max_length, sep_token):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        # Define special tokens
        special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        # Resize token embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.max_length = max_length
        self.sep_token = sep_token

    def forward(self, source, targets=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device) 
        if targets:
            # prepare input
            messages = []
            for i, src in enumerate(source):
                msg = ' '.join([src, self.sep_token, self.tokenizer.bos_token, targets[i], self.tokenizer.eos_token])
                messages.append(msg)
            # tokenize
            model_inputs = self.tokenizer(messages, truncation=True, padding=True, max_length=self.max_length,
                                          return_tensors="pt").to(device)
            labels = self.tokenizer(messages, truncation=True, padding=True, max_length=self.max_length,
                                        return_tensors="pt").input_ids.to(device)
            # Predict
            output = self.model(**model_inputs, labels=labels)  # forward pass
        else:
            # prepare input
            messages = []
            for i, src in enumerate(source):
                msg = ' '.join([src, self.sep_token, self.tokenizer.bos_token])
                messages.append(msg)
            # tokenize
            model_inputs = self.tokenizer(messages, truncation=True, padding=True, max_length=self.max_length,
                                          return_tensors="pt").to(device)
            # Generate texts
            generated_ids = self.model.generate(**model_inputs,
                                                max_length=self.max_length,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                bos_token_id=self.tokenizer.bos_token_id,
                                                # num_beams=2,
                                                length_penalty=1.0,
                                                early_stopping=True)
            # print("generated_ids size:", generated_ids.size())  # Add this line to print generated_ids size
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            output = [w.split(self.sep_token)[-1] for w in output]
        return output