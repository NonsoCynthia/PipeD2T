import os
import json
import torch
import argparse
#import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, Dataset
from data.load_dataset import CustomDataset, preprocess_data, realize_date
from models.BART import BART_Model
from models.GPT2 import GPT2_Model
from models.T5 import T5_Model
from models.LLAMA import LLAMA_Model
from models.FALCON import FALCON_Model
#from torchmetrics.classification import StatScores
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from mapping import entity_mapping, prcs_entry, delist, split_triples


class Inferencer:
    def __init__(self, model, testdata, task, batch_status, write_dir, verbose=True):
        self.model = model
        self.testdata = testdata
        self.task = task
        self.batch_status = batch_status
        self.write_dir = write_dir
        self.verbose = verbose

        if not os.path.exists(write_dir):
            os.mkdir(write_dir)

    def evaluate(self):
        # Evaluate the model's performance
        self.model.model.eval() #sets the model in evaluation mode using eval function
        results = []
        for batch_idx, inputs in enumerate(self.testdata):
            source = inputs.get('Source', None)
            targets = inputs.get('Target', None)
            
            # if source not in results:
            # Initialize the dictionary for this source
            result = {'idx': batch_idx, 'input': source, 'pred': '', 'refs': []} # hyp=generated texts; refs=original targets

            # Predict
            output = self.model([source])
            #result['pred'] = output[0].strip().replace('\n', ' ')
            if self.task == "reg":
                feedback = output[0].strip().replace('\n',' ').replace('[','').replace(']','')
                result['pred'] = feedback[:-1] if feedback.endswith('.') and len(feedback) > 1 else feedback
            else:
                result['pred'] = output[0].strip().replace('\n',' ')


            # Display evaluation progress
            if (batch_idx + 1) % self.batch_status == 0:
                print('Evaluation: [{}/{} ({:.0f}%)]'.format(batch_idx + 1, \
                            len(self.testdata), 100. * batch_idx / len(self.testdata)))

            # Store references as a list of lists
            result['refs'].append(targets)

            # Add the result to the results list
            results.append(result)
            
        # Sort results
        results = sorted(results, key=lambda x: x['idx'])
        # Define result directory path
        path = os.path.join(self.write_dir, f'{self.task}.txt')
        with open(path, 'w') as f:
            f.write('\n'.join([w['pred'] for w in results]))
        
        path = os.path.join(self.write_dir, f'{self.task}.json')
        json.dump(results, open(path, 'w'), separators=(',', ':'), sort_keys=True, indent=4)
       
        #if targets is not None:
            #hyps, refs = [], []
            #for i, row in enumerate(results):
                #try:
                    #hyps.append(nltk.word_tokenize(row['pred']))
                    #print(i, ' '.join([nltk.word_tokenize(ref) for ref in row['refs']]))
                    #refs.append(' '.join([nltk.word_tokenize(ref) for ref in row['refs']]))
                #except Exception as e:
                    #print(f"Error processing row['refs']: {e}")

            #print(len(hyps), len(refs))
            #chencherry = SmoothingFunction()
            #bleu = corpus_bleu(refs, hyps, smoothing_function=chencherry.method3)
            #print(f'BLEU: {bleu}')
            #return bleu

    def evaluate_reg(self):
        self.model.model.eval()

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
                            candidates = self.model([go_in])[0].strip().replace('\n', ' ').replace('[','').replace(']','')
                            candidates = candidates[:-1] if candidates.endswith('.') and len(candidates) > 1 else candidates
                            refex = candidates
                    entry[i] = refex
                    pre_context.append(entity)
                else:
                    pre_context.append(token.lower())

            ent = ' '.join(entry).replace(' .', '.').replace(' ,', ',').replace(' !', '!') + '.'
            results.append(ent)

        # Define result directory path
        path = os.path.join(self.write_dir, f'{self.task}.txt')
        with open(path, 'w') as f:
            f.write('\n'.join([w for w in results])) 

 
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", help="path to the tokenizer")
    parser.add_argument("--model_path", help="path to the model")
    parser.add_argument("--task", help="Traing task")
    parser.add_argument("--data_path", help="path to the data")
    parser.add_argument("--batch_size", help="batch size of test", type=int)
    parser.add_argument("--max_length", help="maximum length to be processed by the network", type=int)
    parser.add_argument("--verbose", help="should display the loss?", action="store_true")
    parser.add_argument("--batch_status", help="display of loss", type=int)
    # parser.add_argument("--cuda", help="use CUDA", action="store_true")
    args = parser.parse_args()   

    # Model settings, Settings and configurations
    tokenizer_path = args.tokenizer
    model_path = args.model_path
    task = args.task
    data_path = args.data_path
    batch_size = args.batch_size
    max_length = args.max_length
    verbose = args.verbose if 'verbose' in args else False
    batch_status = args.batch_status
    # device = torch.device('cuda' if args.cuda else 'cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_path = os.path.join(args.model_path, args.task)


    # Create model
    if 'bart' in tokenizer_path:
        mod = 'bart'
        model_path = os.path.join(model_path, f"{task}/{mod}/model")
        write_path = os.path.join(write_path, mod) 
        generator = BART_Model(tokenizer_path, model_path, max_length, sep_token="<"+task+">")
    elif 't5' in tokenizer_path:
        mod = 'flan-t5-large' if 'flan' in tokenizer_path else 't5-large'
        model_path = os.path.join(model_path, f"{task}/{mod}/model")
        write_path = os.path.join(write_path, mod) 
        generator = T5_Model(tokenizer_path, model_path, max_length, sep_token=task+":")
    elif 'gpt2' in tokenizer_path:
        mod = 'gpt2'
        model_path = os.path.join(model_path, f"{task}/{mod}/model")
        write_path = os.path.join(write_path, mod) 
        generator = GPT2_Model(tokenizer_path, model_path, max_length, sep_token=task+":")
    elif 'falcon' in tokenizer_path:
        mod = 'falcon'
        model_path = os.path.join(model_path, f"{task}/{mod}/model")
        write_path = os.path.join(write_path, mod)
        generator = FALCON_Model(tokenizer_path, model_path, max_length, sep_token=task+":")
    elif 'llama' in tokenizer_path:
        mod = 'llama'
        model_path = os.path.join(model_path, f"{task}/{mod}/model")
        write_path = os.path.join(write_path, mod)
        generator = LLAMA_Model(tokenizer_path, model_path, max_length, sep_token=task+":")
    else:
        raise Exception("Invalid model")


    dataset_dict = preprocess_data(data_path, task, mod)
    # evaluation = {
    #     f"{task}_pipeline_eval": dataset_dict["pipeline_eval"],
    #     f"{task}_pipeline_test": dataset_dict["pipeline_test"]
    # }
    evaluation = {
        f"{task}_dev": dataset_dict["validation"],
        f"{task}_test": dataset_dict["test"],
        f"{task}_pipeline_eval": dataset_dict["pipeline_eval"],
        f"{task}_pipeline_test": dataset_dict["pipeline_test"],  # Note: Assuming this is intentional
    }

    for dataset_name, dataset in evaluation.items():
        #dataset = DataLoader(CustomDataset(dataset), batch_size=1, shuffle=False)
        inf = Inferencer(generator, dataset, dataset_name, batch_status, write_path, verbose)
        print(f"Evaluating {mod} {dataset_name} dataset:")
        if task == "reg" and dataset_name not in [f"{task}_dev", f"{task}_test"]:
            inf.evaluate_reg()
        else:
            inf.evaluate() 
