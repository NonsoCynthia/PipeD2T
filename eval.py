#!/usr/bin/env python3

import json
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from nltk import word_tokenize
from mapping import read_file_map

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

def eval_ord_str(input_path, path_result, write_dir, models, domain_type):
    unseen_domains = ['Artist', 'Politician', 'CelestialBody', 'Athlete', 'MeanOfTransportation']

    result = f'\nEvaluation of pipeline Ordering and Structuring task\n'
    write_file(write_dir, result, mode='a')

    # Store results for each model in a dictionary
    model_results = {model: '' for model in models}
    
    for _set in ['dev', 'test']:
        for task in ['ordering', 'structuring']:
            for domain in domain_type: 
                gold_path = os.path.join(input_path, task, f'{_set}.json')
                gold = read_file(gold_path)

                result = ''
                for model in models:
                    p = os.path.join(path_result, task, model, f'{task}_{_set}.txt')
                    y_predict = [i.replace('[', '<').replace(']', '>') for i in read_file_map(p, task)]

                    y_real, y_pred = [], []
                    for i, item in enumerate(gold):
                        if (domain == "All domains" or 
                            (domain == "Seen domains" and item['category'] not in unseen_domains) or
                            (domain == "Unseen domains" and item['category'] in unseen_domains)):
                            gold_texts = [' '.join(target['output']) for target in item['targets']]
                            y_real.append(gold_texts)
                            y_pred.append(y_predict[i].strip())

                    num, dem = 0.0, 0
                    for i, item in enumerate(y_pred):
                        y = y_real[i]
                        if item.strip() in y:
                            num += 1
                        dem += 1

                    accuracy = round(num / dem, 2) if dem != 0 else 0.0

                    result += '\n' + 'Task: ' + task
                    result += '\n' + 'Set: ' + _set
                    result += '\n' + 'Model: ' + model
                    result += '\n' + 'Domain: ' + domain
                    result += '\n' + 'Accuracy: ' + str(accuracy)
                    result += '\n' + 20 * '-' + '\n'

                    # Append results to the model's section
                model_results[model] += result
    
    # Write results for each model
    for model, result in model_results.items():
        write_file(write_dir, result, mode='a')        
    write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')

  
def eval_lex(input_path, path_result, write_dir, models, domain_type):
    unseen_domains = ['Artist', 'Politician', 'CelestialBody', 'Athlete', 'MeanOfTransportation']
    task = 'lexicalization' 
    result = f'\nEvaluation of Lexicalization task\n'
    write_file(write_dir, result, mode='a')

    # Store results for each model in a dictionary
    model_results = {model: '' for model in models}

    for _set in ['dev', 'test']:
        for domain in domain_type:
            gold_path = os.path.join(input_path, task, f'{_set}.json')  # path/task/set.json
            gold = read_file(gold_path)

            result = ''
            for model in models:
                p = os.path.join(path_result, task, model, f'{task}_{_set}.txt')
                y_predict = [i.replace('[', '<').replace(']', '>') for i in read_file_map(p, task)]

                y_real, y_pred = [], []
                for i, item in enumerate(gold):
                    if (domain == "All domains" or 
                        (domain == "Seen domains" and item['category'] not in unseen_domains) or
                        (domain == "Unseen domains" and item['category'] in unseen_domains)):
                        gold_texts = [' '.join(target['output']).lower() for target in item['targets']]
                        y_real.append(gold_texts)
                        y_pred.append(y_predict[i].strip().lower())

                # Tokenize y_real and y_pred
                y_real_tokenized = [[nltk.word_tokenize(sent) for sent in ref] for ref in y_real]
                y_pred_tokenized = [nltk.word_tokenize(sent) for sent in y_pred]

                # Calculate BLEU score
                chencherry = SmoothingFunction()
                try:
                    bleu_score = corpus_bleu(y_real_tokenized, y_pred_tokenized, smoothing_function=chencherry.method3)
                except:
                    bleu_score = 0

                result += '\n' + 'Task: '+ "Lexicalization"
                result += '\n' + 'Set: '+ _set 
                result += '\n' + 'Model: '+ model 
                result += '\n' + 'Domain: ' + domain
                result += '\n' + 'Result: '+ str(round(bleu_score, 2))
                result += '\n' + 20 * '-' + '\n'

            # Append results to the model's section
            model_results[model] += result
    
     # Write results for each model
    for model, result in model_results.items():
        write_file(write_dir, result, mode='a')
        
    write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')


def eval_reg(input_path, path_result, write_dir, models, domain_type):
    unseen_domains = ['Artist', 'Politician', 'CelestialBody', 'Athlete', 'MeanOfTransportation']
    task = 'reg'
    result = f'\nEvaluation of pipeline {task} task\n'
    write_file(write_dir, result, mode='a')

    # Store results for each model in a dictionary
    model_results = {model: '' for model in models}

    for _set in ['dev', 'test']:
        for domain in domain_type:
            gold_path = os.path.join(input_path, task, f'{_set}.json')  # path/task/set.json
            gold = read_file(gold_path)

            result = ''
            for model in models:
                p = os.path.join(path_result, task, model, f'{task}_{_set}.txt')
                y_predict = [i.replace('[', '<').replace(']', '>') for i in read_file_map(p, task)]

                y_real, y_pred = [], []
                for i, item in enumerate(gold):
                    refex_str = ' '.join(item['refex']).strip().lower()
                    y_real.append(' '.join(nltk.word_tokenize(refex_str)))
                    y_pred.append(y_predict[i].strip().lower())

                    num, dem = 0.0, 0
                    baseline = 0
                for i, item in enumerate(y_pred):
                    if (domain == "All domains" or 
                        (domain == "Seen domains" and gold[i]['category'] not in unseen_domains) or
                        (domain == "Unseen domains" and gold[i]['category'] in unseen_domains)):
                        entity_str = str(gold[i]['entity']).replace('\'', ' ').replace('\"', ' ').replace('_', ' ')
                        refex = ' '.join(nltk.word_tokenize(entity_str.lower()))            
                        y = y_real[i]
                        if item.strip() == y:
                            num +=1
                            # print(item.strip())
                        if refex.strip().lower() == y:
                            baseline += 1
                            # print(item.strip().lower())
                        dem += 1

                result += '\n' + 'Task: '+ "REG"
                result += '\n' + 'Set: '+ _set 
                result += '\n' + 'Model: '+ model
                result += '\n' + 'Domain: ' + domain
                result += '\n' + 'Baseline Accuracy: '+ str(round(baseline/dem, 2) if dem > 0 else 0)
                result += '\n' + 'Accuracy: '+ str(round(num/dem, 2) if dem > 0 else 0)
                result += '\n' + 20 * '-' + '\n'
    
            # Append results to the model's section
            model_results[model] += result
    
     # Write results for each model
    for model, result in model_results.items():
        write_file(write_dir, result, mode='a')
        
    write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')


def eval_bleu_sr(input_path, path_result, write_dir, models, domain_type):
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

                    y_real, y_pred = [], []
                    for i, item in enumerate(gold):
                        if (domain == "All domains" or 
                            (domain == "Seen domains" and item['category'] not in unseen_domains) or
                            (domain == "Unseen domains" and item['category'] in unseen_domains)):
                            gold_texts = [' '.join(target['output']).lower() for target in item['targets']]
                            y_real.append(gold_texts)
                            y_pred.append(y_predict[i].strip().lower())
            
                    y_real_tokenized = [[nltk.word_tokenize(sent) for sent in ref] for ref in y_real]
                    y_pred_tokenized = [nltk.word_tokenize(sent) for sent in y_pred]

                    chencherry = SmoothingFunction()
                    try:
                        bleu_score = corpus_bleu(y_real_tokenized, y_pred_tokenized, smoothing_function=chencherry.method3)
                    except:
                        bleu_score = 0

                    result = '\n' + 'Task: ' + task
                    result += '\n' + 'Set: ' + _set
                    result += '\n' + 'Model: ' + model
                    result += '\n' + 'Domain: ' + domain
                    result += '\n' + 'Result_bleu: ' + str(round(bleu_score, 2))
                    result += '\n' + 20 * '-' + '\n'
    
                    # Append results to the model's section
                    model_results[model] += result
    
    # Write results for each model
    for model, result in model_results.items():
        write_file(write_dir, result, mode='a')
        
    write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')


def eval_meteor_sr(input_path, path_result, write_dir, models, domain_type):
    unseen_domains = ['Artist', 'Politician', 'CelestialBody', 'Athlete', 'MeanOfTransportation']
    result = f'\nEvaluation of Meteor Score for pipeline End2end and Surface Realization task\n'
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

                    y_real, y_pred = [], []
                    for i, item in enumerate(gold):
                        if (domain == "All domains" or 
                            (domain == "Seen domains" and item['category'] not in unseen_domains) or
                            (domain == "Unseen domains" and item['category'] in unseen_domains)):
                            targets = [' '.join(target['output']) for target in item['targets']]
                            t = [' '.join(target).lower() for target in targets]
                            y_real.append(t)
                            pred = ' '.join(y_predict[i].strip()).lower()
                            y_pred.append(pred)
            
                    # Calculate METEOR score
                    try:
                        meteor = 0
                        for i in range(len(y_real)):
                            # Convert list of tokenized sentences to a single string
                            ref_str = ' '.join([''.join(sent) for sent in y_real[i]])
                            ref_str = nltk.word_tokenize(ref_str)
                            hypothesis_tokens = nltk.word_tokenize(y_pred[i])
                            meteor += single_meteor_score(ref_str , hypothesis_tokens)
                        meteor_ = meteor/len(y_real)
                    except ZeroDivisionError:
                        meteor_ = 0.0

                    result = '\n' + 'Task: ' + task
                    result += '\n' + 'Set: ' + _set
                    result += '\n' + 'Model: ' + model
                    result += '\n' + 'Domain: ' + domain
                    result += '\n' + 'Result_Meteor: ' + str(round(meteor_, 2))
                    result += '\n' + 20 * '-' + '\n'
    
                    # Append results to the model's section
                    model_results[model] += result
    
    # Write results for each model
    for model, result in model_results.items():
        write_file(write_dir, result, mode='a')
        
    write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')


if __name__ == "__main__":

        # original_path = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/"
        original_path = ''
        input_path = original_path + 'data/deepnlg/input/'
        path_result = original_path + "results/"
        write_dir = path_result+"eval_test_results.txt"

        # models = ["t5", "t5-large", "bart", "gpt2", "gpt-3.5"]
        models = ["t5", "bart", "gpt2"]
        domain_type = ["All domains", "Seen domains", "Unseen domains"]

        heading = "Evaluation of Pipeline Neural Architecture Using Webnlg dataset on LLM's"
        write_file(write_dir, heading, mode='w')

        #Ordering and Structuring
        eval_ord_str(input_path, path_result, write_dir, models, domain_type)

        # Lexicalization
        eval_lex(input_path, path_result, write_dir, models, domain_type)

        # Referring Expression Generation
        eval_reg(input_path, path_result, write_dir, models, domain_type)

        # Surface Realization and End2end generation
        eval_bleu_sr(input_path, path_result, write_dir, models, domain_type) #Bleu
        # eval_meteor_sr(input_path, path_result, write_dir, models, domain_type) #Meteor

        # ('t5-base' if model == 't5' else model)
        print("Evaluation Finished!!!!!")