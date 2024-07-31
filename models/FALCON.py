import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, set_seed

set_seed(42)

class FALCON_Model(nn.Module):
    def __init__(self, tokenizer_path, model_path, max_length, sep_token):
        super().__init__()
        self.max_length = max_length
        self.sep_token = sep_token
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        tokenizer.pad_token=tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                trust_remote_code=True,
                                                load_in_4bit=True,
                                                device_map='auto'
                                                )

    def forward(self, source, targets=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            # Predict
            output = self.model(**model_inputs, labels=labels) # forward pass
        else:
            generated_ids = self.model.generate(**model_inputs,
                                                   max_length=self.max_length,
                                                   num_beams=4,
                                                   no_repeat_ngram_size=3,
                                                   do_sample=True,
                                                   #temperature=0.9,
                                                   top_p=0.9,
                                                   repetition_penalty=2.0,
                                                   early_stopping=True
                                                   )
            output = self.tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)
        return output 
