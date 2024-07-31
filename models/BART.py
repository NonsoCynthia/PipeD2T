import torch
import torch.nn as nn
#import pytorch_lightning as pl
from transformers import BartForConditionalGeneration, BartTokenizerFast as BartTokenizer, set_seed, GenerationConfig
#pl.seed_everything(42)

set_seed(42)

class BART_Model(nn.Module):
    def __init__(self, tokenizer_path, model_path, max_length, sep_token):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.max_length = max_length
        self.sep_token = sep_token
        
        # Define non-default generation parameters
        generation_config = GenerationConfig(
                                            early_stopping=True,
                                            do_sample=True,
                                            top_p=0.95,
                                            top_k=10,
                                            num_beams=4,
                                            no_repeat_ngram_size=3,
                                            repetion_penalty=2.0,
                                            )

        # Set the generation config for the model
        #self.model.config.generation = generation_config
    
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
                                    #)        
                                    early_stopping=True,
                                    do_sample=True,
                                    #top_p=0.95,
                                    #top_k=10,
                                    num_beams=4,
                                    #temperature=0.9,
                                    no_repeat_ngram_size=3,
                                    repetition_penalty=2.0,
                                    )
   
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return output
