import torch 
import torch.nn as nn 
import pytorch_lightning as pl 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, BitsAndBytesConfig  
 
pl.seed_everything(42) 
 
class LLAMA_Model(pl.LightningModule): 
    def __init__(self, tokenizer_path, model_path, max_length, sep_token): 
        super().__init__() 
        self.max_length = max_length 
        self.sep_token = sep_token 
        self.hf_auth = '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'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Add a padding token to the tokenizer 
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        self.model_config = AutoConfig.from_pretrained(model_path, token=self.hf_auth) 
        self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                config=self.model_config,
                                                trust_remote_code=True,
                                                token=self.hf_auth,
                                                load_in_4bit=True,
                                                ) 
        

    def forward(self, source, targets=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # prepare
        for i, src in enumerate(source):
            prepared_source = ' '.join([self.sep_token, src])
            source[i] = prepared_source
        # tokenize
        model_inputs = self.tokenizer(source, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(device).to(torch.long) 
        # Predict
        self.model = self.model.to(device) 
        if targets:
            labels = self.tokenizer(targets, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").input_ids.to(device).to(torch.long) 
            # Predict
            output = self.model(**model_inputs, labels=labels) # forward pass
        else:
            # generated_ids = self.model.generate(**model_inputs, 
            #                                     max_length=self.max_length,
            #                                     # num_beams=2,
            #                                     # length_penalty=1.0,
            #                                     # early_stopping=True
            #                                     )
            with torch.no_grad(): 
                generated_ids = self.model.generate(model_inputs.to(torch.long), do_sample=True, max_new_tokens=self.max_length, num_beams=2, length_penalty=1.0, temperature=1.0) 
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output
