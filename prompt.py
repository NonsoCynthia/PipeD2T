import os
import re
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader
from data.load_dataset import CustomDataset, preprocess_data, realize_date, read_file
from dotenv import load_dotenv, find_dotenv
from instruction import *
from mapping import entity_mapping, prcs_entry, delist, split_triples

_ = load_dotenv(find_dotenv())  # read local .env file
hf_token = os.getenv("HF_TOKEN")


# Function to write files
def write_file(write_path, result, mode='w'):
    with open(write_path, mode) as f:
        f.write('\n'.join(result))

def extract_middle_text(triple_string, task):
    if task in ["ordering", "structuring"]:
        tr = ' '.join(triple_string).replace("[TRIPLE]", '').strip()
        g  = tr.split("[/TRIPLE]")
        pred = []
        for i in g:
            # Split the triple into a list of words
            words = i.strip().split()
            print(words)
            if len(words) >= 2:
                # Extract the middle text if there are at least two words
                b = words[1]
                pred.append(b)
        # Return the list of middle texts if there is more than one
        if len(pred) == 1:
            return pred[0] if task == "ordering" else f"[SNT] {pred[0]} [/SNT]"
        else:
            return None
    else:
        return None


class LlamaModel(torch.nn.Module):
    def __init__(self, model_id, hf_auth, max_length=512):
        super(LlamaModel, self).__init__()
        self.max_length = max_length
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_auth)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = self.load_model(model_id, hf_auth)

    def load_model(self, model_id, hf_auth):
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type='nf4',
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_compute_dtype=torch.bfloat16
                                        )
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     quantization_config=bnb_config,
                                                     trust_remote_code=True,
                                                     token=hf_auth,
                                                     #use_safetensors=True,
                                                     device_map="auto"
                                                     )
        return model

    def forward(self, source, targets=None):
        # Format message with the command-r chat template
        messages = [{"role": "user", "content": source}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        #input_ids = self.tokenizer.encode(source, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(input_ids.to(torch.long), pad_token_id=self.tokenizer.eos_token_id, do_sample=True, 
                                                 max_new_tokens=300, num_beams=2, repetition_penalty=2.0)
            prompt_len = input_ids.shape[-1] #[prompt_len:]
            generated_text = self.tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True).strip()      
            generated_text = re.sub('\n+', '\n', generated_text)  # remove excessive newline characters
        return generated_text

class Inferencer:
    def __init__(self, model, prompt, testdata, task, write_dir):
        self.model = model
        self.prompt = prompt
        self.testdata = testdata
        self.task = task
        self.write_dir = write_dir

    def evaluate(self):
        # Evaluate the model's performance
        self.model.eval()
        results = []
        for batch_idx, inputs in enumerate(self.testdata):
            source = inputs.get('Source', None)
            if source:
                # Predict
                #print(source)
                #one_triple = extract_middle_text(source, self.task)
                #if one_triple:
                    #result = one_triple
                #else:
                #prompt_in = instruct_templates('mistral', source, 'struct2sr', pipeline=True)
                prompt_in = instruct_templates('mistral', source, task, pipeline=False)
                output = self.model(prompt_in)
                result = output.split(prompt_in)[-1].strip().replace('\n', '  ')
                print(batch_idx,result)
                results.append(result)
            else:
                print("No Source")

        # Write the results into the path
        write_file(self.write_dir, results, mode='w')
        print(f'{self.write_dir} Ended!!!!', "\n")

    def evaluate_reg(self):
        self.model.eval()

        source = self.testdata['Source']
        targets = self.testdata['Target']

        source = [split_triples(t.split()) for t in source]
        entity_maps = [entity_mapping(t) for t in source]
        entity_maps = [dict((k, v.replace('_', ' ').replace('\"', ' ').replace('\'', ' ').strip()) for k, v in
                            entity_mapping(t).items()) for t in source]

        results = []  # Rename the inner results list to results

        for y, entry in enumerate(targets):
            pre_context = []
            entry = prcs_entry(entry).split()

            for i, token in enumerate(entry):
                if prcs_entry(token) in entity_maps[y]:
                    entity = entity_maps[y][prcs_entry(token)]
                    isDate, refex = realize_date(entity)
                    if not isDate:
                        try:
                            refex = str(int(entity))
                        except ValueError:
                            pos_context = []  # Reset pos_context for each entity
                            for j in range(i + 1, len(entry)):
                                clean_token = prcs_entry(entry[j]).strip()
                                # ctoken = clean_token = clean_token.replace('.', '').strip()
                                if clean_token in entity_maps[y]:
                                    # pos_context.append(entity_maps[y][clean_token.strip('.')])
                                    pos_context.append(entity_maps[y][clean_token])
                                else:
                                    pos_context.append(clean_token.lower())
                            go_in = (delist(pre_context) + ' ' + delist(pos_context) + '. ' + entity).strip()
                            print(f"go_in{i}: {go_in}")
                            prompt_in = instruct_templates('mistral', go_in, task, pipeline=False)
                            result = self.model(prompt_in)
                            result = result.strip().replace('\n', ' ').replace('[','').replace(']','').replace("#","")
                            result = result[:-1] if result.endswith('.') and len(result) > 1 else result
                            #assign the Inference to the REG position in the text
                            refex = result.strip()
                    entry[i] = refex
                    pre_context.append(entity)
                else:
                    pre_context.append(token.lower())

            ent = ' '.join(entry).replace(' .', '.').replace(' ,', ',').replace(' !', '!')
            print(f"Input {y}: {ent}")
            results.append(ent)

        # Define result directory path
        write_file(self.write_dir, results, mode='w')
        print(f"{self.write_dir} Ended!!!!", '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to the model")
    parser.add_argument("--model_path", help="path to save the results")
    parser.add_argument("--task", help="Training task")
    parser.add_argument("--data_path", help="path to the data")
    parser.add_argument("--write_path", help="path to write best model")
    args = parser.parse_args()

    task = args.task
    data_path = args.data_path
    model_id = args.model
    model_path = args.model_path

    write_path = os.path.join(f'{args.write_path}', f'{args.task}', f'{model_path}')  #f'{model_path}_struct')
    # Create result directory if it doesn't exist.
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    dataset_dict = preprocess_data(data_path, task, model_path)
    evaluation = {
        #f"{task}_dev": dataset_dict["validation"],
        #f"{task}_test": dataset_dict["test"],
        f"{task}_pipeline_eval": dataset_dict["pipeline_eval"],
        f"{task}_pipeline_test": dataset_dict["pipeline_test"]
    }

    prompt_dir = os.path.join(f'{args.write_path}', f'{args.task}', f'{task}_prompts.txt')
    prompt = open(prompt_dir, "r").read() #read_file(prompt_dir)
    #print(prompt)

    hf_auth = hf_token
    max_length = 1024
    llama_model = LlamaModel(model_id, hf_auth, max_length)

    # # Feed the chatgpt the dev, test, and pipeline datasets for inference
    for dataset_name, dataset in evaluation.items():
        print(f"Evaluating {model_path} {dataset_name}.txt dataset:")
        path = os.path.join(write_path, f'{dataset_name}.txt')                     
        #dataset = DataLoader(CustomDataset(dataset), batch_size=1, shuffle=False) #num_workers=10
        inf = Inferencer(llama_model, prompt, dataset, task, path)
        if task == "reg" and dataset_name not in [f"{task}_dev", f"{task}_test"]:
            inf.evaluate_reg()
        else:
            inf.evaluate()
        print(f'{dataset_name}.txt Ended!!!!', "\n")

    t_task = {
    "ordering": "order",
    "structuring": "structure",
    "lexicalization": "lexicalized format",
    "reg": "refering expression generation format",
    "sr": "textual realization format"
    }

