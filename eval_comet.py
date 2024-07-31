#### !/usr/bin/env python3

import json
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from nltk import word_tokenize
from mapping import read_file_map
from comet import download_model, load_from_checkpoint
#from evaluate import load
#comet_metric = load('comet')

def read_file(path):
    if path.endswith(".json"):
        with open(path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    else:
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.read()
            contents = contents.replace('[', '<').replace(']', '>')
            lines = [line.strip() for line in contents.split('\n')]
            if lines and lines[-1] == '':
                return lines[:-1]
            return lines

def write_file(write_path, result, mode='w'):
    with open(write_path, mode) as f:
        f.write(result) 

def eval_comet(input_path, path_result, write_dir, models, domain_type, comet_model):
    unseen_domains = ['Artist', 'Politician', 'CelestialBody', 'Athlete', 'MeanOfTransportation']

    result = f'\nEvaluation of Bleu Score for pipeline End2end and Surface Realization task\n'
    write_file(write_dir, result, mode='a')

    # Store results for each model in a dictionary
    model_results = {model: '' for model in models}
    
    for _set in ['dev', 'test']:
        for task in ['end2end', 'sr']:
            for domain in domain_type:
                gold_path = os.path.join(input_path, 'end2end', f'{_set}.json')  # path/task/set.json
                gold = read_file(gold_path)

                for model in models:
                    # _set = 'eval' if _set == 'dev' else _set
                    sw = {"dev":"eval", "test":"test"}
                    p = os.path.join(path_result, task, model, f'{task}_pipeline_{sw[_set]}.txt')
                    y_predict = read_file_map(p, task)

                    extracted_data = []
                    y_src, y_real, y_pred = [], [], []
                    for i, item in enumerate(gold):
                        if (domain == "All domains" or 
                            (domain == "Seen domains" and item['category'] not in unseen_domains) or
                            (domain == "Unseen domains" and item['category'] in unseen_domains)):
                            # Join source tokens excluding <TRIPLE> and </TRIPLE>
                            gold_source = ' '.join(item['source']).replace('<','[').replace('>',']')
                            y_src.append(gold_source)
                            gold_texts = [' '.join(target['output']).lower() for target in item['targets']]
                            y_real.append(gold_texts)
                            y_pred.append(y_predict[i].strip().lower())

                            extracted_data.append({'src':gold_source, 'mt':y_predict[i], 'ref':gold_texts})


                    try:
                        comet_score = comet_model.predict(extracted_data, batch_size=1, gpus=1)
                        #comet_score = sum(comet_scores.system_score) / len(comet_scores.system_score)
                        print(f'comet_score: {comet_score}')
                        print(f'comet_score.scores:{comet_score.scores}') # sentence-level scores
                        print(f'comet_score.system_score:{comet_score.system_score}') # system-level score
                    except:
                        comet_score = 0

                    result = '\n' + 'Task: ' + task
                    result += '\n' + 'Set: ' + _set
                    result += '\n' + 'Model: ' + model
                    result += '\n' + 'Domain: ' + domain
                    #result += '\n' + 'Result_bleu: ' + str(round(comet_score, 2))
                    result += '\n' + 20 * '-' + '\n'
    
                    # Append results to the model's section
                    model_results[model] += result
    
    # Write results for each model
    for model, result in model_results.items():
        write_file(write_dir, result, mode='a')
        
    write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')


from evaluate import load
comet_metric = load('comet') 

def eval_comet_hf(input_path, path_result, write_dir, models, domain_type, comet_metric):
    unseen_domains = ['Artist', 'Politician', 'CelestialBody', 'Athlete', 'MeanOfTransportation']

    result = f'\nEvaluation of Bleu Score for pipeline End2end and Surface Realization task\n'
    write_file(write_dir, result, mode='a')

    # Store results for each model in a dictionary
    model_results = {model: '' for model in models}
    
    for _set in ['dev', 'test']:
        for task in ['end2end', 'sr']:
            for domain in domain_type:
                gold_path = os.path.join(input_path, 'end2end', f'{_set}.json')  # path/task/set.json
                gold = read_file(gold_path)

                for model in models:
                    # _set = 'eval' if _set == 'dev' else _set
                    sw = {"dev":"eval", "test":"test"}
                    p = os.path.join(path_result, task, model, f'{task}_pipeline_{sw[_set]}.txt')
                    y_predict = read_file_map(p, task)

                    y_src, y_real, y_pred = [], [], []
                    for i, item in enumerate(gold):
                        if (domain == "All domains" or 
                            (domain == "Seen domains" and item['category'] not in unseen_domains) or
                            (domain == "Unseen domains" and item['category'] in unseen_domains)):
                            # Join source tokens excluding <TRIPLE> and </TRIPLE>
                            gold_source = ' '.join(item['source']).replace('<','[').replace('>',']')
                            y_src.append(gold_source)
                            gold_texts = [' '.join(target['output']).lower() for target in item['targets']]
                            y_real.append(gold_texts)
                            y_pred.append(y_predict[i].strip().lower())
            
                    # y_src_tokenized = [nltk.word_tokenize(sent) for sent in y_src]
                    # y_real_tokenized = [[nltk.word_tokenize(sent) for sent in ref] for ref in y_real]
                    # y_pred_tokenized = [nltk.word_tokenize(sent) for sent in y_pred]


                    try:
                        score = 0
                        for i in range(len(y_pred)):
                            comet_sc = comet_metric.compute(predictions=y_pred[i], references=y_real[i], sources=y_src[i])
                            score += sum(comet_sc)
                        comet_score = score/len(y_pred)
                    except:
                        comet_score = 0

                    result = '\n' + 'Task: ' + task
                    result += '\n' + 'Set: ' + _set
                    result += '\n' + 'Model: ' + model
                    result += '\n' + 'Domain: ' + domain
                    result += '\n' + 'Result_bleu: ' + str(round(comet_score, 2))
                    result += '\n' + 20 * '-' + '\n'
    
                    # Append results to the model's section
                    model_results[model] += result
    
    # Write results for each model
    for model, result in model_results.items():
        write_file(write_dir, result, mode='a')
        
    write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')


if __name__ == "__main__":

    original_path = ''
    input_path = original_path + 'data/deepnlg/input/'
    path_result = original_path + "results/"
    write_dir = "comet_results.txt"
    models = ["t5", "bart", "gpt2"]
    domain_type = ["All domains", "Seen domains", "Unseen domains"]
    model_path = download_model("Unbabel/XCOMET-XL")
    comet_model = load_from_checkpoint(model_path)
    #eval_comet(input_path, path_result, write_dir, models, domain_type, comet_model)
    eval_comet_hf(input_path, path_result, write_dir, models, domain_type, comet_metric)
