import json
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
import numpy as np
import subprocess
import re
from mapping import read_file_map
# nltk.download('punkt')
# nltk.download('wordnet')

# Function to read json and .txt files
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

# Function to write files
def write_file(write_path, result, mode='w'):
    with open(write_path, mode) as f:
        f.write(result)

# root path
# original_path = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/"
original_path = ''
path_input = original_path + 'data/deepnlg/input/'
path_result = original_path + "results/"
write_dir = path_result+"eval_results.txt"

################################################
    #  Evaluation of Ordering and Structuring #
################################################
print('Evaluation of Ordering and Structuring')
print('All domains')
Result = "EVALUATION RESULTS FOR DATA-TO-TEXT WEBNLG \n"
result = Result + '\nEvaluation of Ordering and Structuring \n' 
result += '\nAll domains:  \n' 
for _set in ['test', 'dev']:
    for task in ['ordering', 'structuring']:
        gold_path=os.path.join(path_input, task, _set + '.json') #path/task/set.json
        gold = read_file(gold_path)
      
        for model in ["t5", "bart", "gpt2", "chatgpt"]:
            p = os.path.join(path_result, task, model, task+'_'+_set +'.txt')
            y_predict = [i.replace('[', '<').replace(']', '>') for i in read_file_map(p)]
                
                
            y_real, y_pred = [], []
            for i, item in enumerate(gold):
                gold_texts = [' '.join(target['output']) for target in item['targets']]
                y_real.append(gold_texts)
                y_pred.append(y_predict[i].strip())

            num, dem = 0.0, 0
            for i, item in enumerate(y_pred):
                y = y_real[i]
                if item.strip() in y:
                    num +=1
                dem +=1

            result += '\n' + 'Task: '+ task
            result += '\n' + 'Set: '+ _set 
            result += '\n' + 'Model: '+ model 
            result += '\n' + 'Accuracy: '+ str(round(num/dem, 2))
            result += '\n' + 20 * '-' + '\n'
        # result += '\n' + 40 * '-' + '\n' 

write_file(write_dir, result, mode='w')
write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a') 

unseen_domains = ['Artist', 'Politician', 'CelestialBody', 'Athlete', 'MeanOfTransportation']
print('Seen domains')
result = '\nSeen domains:  \n' 
for _set in ['test', 'dev']:
    for task in ['ordering', 'structuring']:
        gold_path=os.path.join(path_input, task, _set + '.json') #path/task/set.json
        gold = read_file(gold_path)
      
        for model in ["t5", "bart", "gpt2", "chatgpt"]:
            p = os.path.join(path_result, task, model, task+'_'+_set +'.txt')
            y_predict = [i.replace('[', '<').replace(']', '>') for i in read_file_map(p)]
                
                
            y_real, y_pred = [], []
            for i, item in enumerate(gold):
                if item['category'] not in unseen_domains:
                    gold_texts = [' '.join(target['output']) for target in item['targets']]
                    y_real.append(gold_texts)
                    y_pred.append(y_predict[i].strip())

            num, dem = 0.0, 0
            for i, item in enumerate(y_pred):
                y = y_real[i]
                if item.strip() in y:
                    num +=1
                dem +=1

            result += '\n' + 'Task: '+ task
            result += '\n' + 'Set: '+ _set 
            result += '\n' + 'Model: '+ model 
            result += '\n' + 'Accuracy: '+ str(round(num/dem, 2))
            result += '\n' + 20 * '-' + '\n'
        # result += '\n' + 40 * '-' + '\n' 

write_file(write_dir, result, mode='a')
write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a') 

print('Unseen domains')
result = '\nUnseen domains:  \n' 
for _set in ['test', 'dev']:
    for task in ['ordering', 'structuring']:
        gold_path=os.path.join(path_input, task, _set + '.json') #path/task/set.json
        gold = read_file(gold_path)
      
        for model in ["t5", "bart", "gpt2", "chatgpt"]:
            p = os.path.join(path_result, task, model, task+'_'+_set +'.txt')
            y_predict = [i.replace('[', '<').replace(']', '>') for i in read_file_map(p)]
                
                
            y_real, y_pred = [], []
            for i, item in enumerate(gold):
                if item['category'] in unseen_domains:
                    gold_texts = [' '.join(target['output']) for target in item['targets']]
                    y_real.append(gold_texts)
                    y_pred.append(y_predict[i].strip())

            num, dem = 0.0, 0
            for i, item in enumerate(y_pred):
                y = y_real[i]
                if item.strip() in y:
                    num +=1
                dem +=1

            result += '\n' + 'Task: '+ task
            result += '\n' + 'Set: '+ _set 
            result += '\n' + 'Model: '+ model 
            result += '\n' + 'Accuracy: '+ str(round(num/dem, 2) if dem > 0 else 0)
            result += '\n' + 20 * '-' + '\n'
        # result += '\n' + 40 * '-' + '\n' 

write_file(write_dir, result, mode='a')
write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')

################################################
    #  Evaluation of Referring Expressions #
################################################

print('Evaluation of Referring Expressions')
print('All domains')
result = '\n\nEvaluation of Referring Expressions \n' 
result += '\nAll domains:  \n' 
 
for _set in ['test', 'dev']:
    gold_path=os.path.join(path_input, 'reg', _set + '.json') #path/task/set.json
    gold = read_file(gold_path)
    
    for model in ["t5", "bart", "gpt2"]:
        p = os.path.join(path_result, 'reg', model, 'reg'+'_'+_set +'.txt')
        # p=/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/reg/t5/reg_test.txt
        y_predict = [i.replace('[', '<').replace(']', '>') for i in read_file_map(p)]
               
        y_real, y_pred = [], []
        for i, item in enumerate(gold):
            # y_real.append(' '.join(i.strip().lower()) for i in item['refex'])
            refex_str = ' '.join(item['refex']).strip().lower()
            y_real.append(' '.join(nltk.word_tokenize(refex_str)))
            y_pred.append(y_predict[i].strip().lower())

        num, dem = 0.0, 0
        baseline = 0
        for i, item in enumerate(y_pred):
            # refex = ' '.join(nltk.word_tokenize(gold[i]['entity'].replace('\'', ' ').replace('\"', ' ').replace('_', ' ')))
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

        result += '\n' +  'Task: '+ "REG"
        result += '\n' + 'Set: '+ _set 
        result += '\n' + 'Model: '+ model 
        result += '\n' + 'Baseline Accuracy: '+ str(round(baseline/dem, 2) if dem > 0 else 0)
        result += '\n' + 'Accuracy: '+ str(round(num/dem, 2) if dem > 0 else 0)
        result += '\n' + 20 * '-' + '\n'
        # result += '\n' + 40 * '-' + '\n' 

write_file(write_dir, result, mode='a')
write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a') 


print('Seen domains')
result = '\nSeen domains:  \n' 
 
for _set in ['test', 'dev']:
    gold_path=os.path.join(path_input, 'reg', _set + '.json') #path/task/set.json
    gold = read_file(gold_path)
    
    for model in ["t5", "bart", "gpt2"]:
        p = os.path.join(path_result, 'reg', model, 'reg'+'_'+_set +'.txt')
        # p=/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/reg/t5/reg_test.txt
        y_predict = [i.replace('[', '<').replace(']', '>') for i in read_file_map(p)]
               
        y_real, y_pred = [], []
        for i, item in enumerate(gold):
            # y_real.append(' '.join(i.strip().lower()) for i in item['refex'])
            refex_str = ' '.join(item['refex']).strip().lower()
            y_real.append(' '.join(nltk.word_tokenize(refex_str)))
            y_pred.append(y_predict[i].strip().lower())

        num, dem = 0.0, 0
        baseline = 0
        for i, item in enumerate(y_pred):
            if gold[i]['category'] not in unseen_domains:
                # refex = ' '.join(nltk.word_tokenize(gold[i]['entity'].replace('\'', ' ').replace('\"', ' ').replace('_', ' ')))
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
        result += '\n' + 'Baseline Accuracy: '+ str(round(baseline/dem, 2) if dem > 0 else 0)
        result += '\n' + 'Accuracy: '+ str(round(num/dem, 2) if dem > 0 else 0)
        result += '\n' + 20 * '-' + '\n'
        # result += '\n' + 40 * '-' + '\n' 

write_file(write_dir, result, mode='a')
write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a') 


print('Unseen domains')
result = '\nUnseen domains:  \n' 
 
for _set in ['test', 'dev']:
    gold_path=os.path.join(path_input, 'reg', _set + '.json') #path/task/set.json
    gold = read_file(gold_path)
    
    for model in ["t5", "bart", "gpt2"]:
        p = os.path.join(path_result, 'reg', model, 'reg'+'_'+_set +'.txt')
        # p=/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/reg/t5/reg_test.txt
        y_predict = [i.replace('[', '<').replace(']', '>') for i in read_file_map(p)]
               
        y_real, y_pred = [], []
        for i, item in enumerate(gold):
            # y_real.append(' '.join(i.strip().lower()) for i in item['refex'])
            refex_str = ' '.join(item['refex']).strip().lower()
            y_real.append(' '.join(nltk.word_tokenize(refex_str)))
            y_pred.append(y_predict[i].strip().lower())

        num, dem = 0.0, 0
        baseline = 0
        for i, item in enumerate(y_pred):
            if gold[i]['category'] in unseen_domains:
                # refex = ' '.join(nltk.word_tokenize(gold[i]['entity'].replace('\'', ' ').replace('\"', ' ').replace('_', ' ')))
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

        result += '\n' +  'Task: '+ "REG"
        result += '\n' + 'Set: '+ _set 
        result += '\n' + 'Model: '+ model 
        result += '\n' + 'Baseline Accuracy: '+ str(round(baseline/dem, 2) if dem > 0 else 0)
        result += '\n' + 'Accuracy: '+ str(round(num/dem, 2) if dem > 0 else 0)
        result += '\n' + 20 * '-' + '\n'
        # result += '\n' + 40 * '-' + '\n' 

write_file(write_dir, result, mode='a')
write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a') 


################################################
    #  Evaluation of Lexicalization #
################################################
print('Evaluation of Lexicalization')
print('All domains')
result = '\n\nEvaluation of Lexicalization \n' 
result += '\nAll domains:  \n' 

for _set in ['test', 'dev']:
    gold_path=os.path.join(path_input, 'lexicalization', _set + '.json') #path/task/set.json
    gold = read_file(gold_path)
    
    for model in ["t5", "bart", "gpt2"]:
        p = os.path.join(path_result, 'lexicalization', model, 'lexicalization'+'_'+_set +'.txt')
        # p=/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/reg/t5/reg_test.txt
        y_predict = [i.replace('[', '<').replace(']', '>') for i in read_file_map(p)]
               
        y_real, y_pred = [], []
        for i, g in enumerate(gold):
        #     if g['category'] in unseen_domains:
            t = [' '.join(target['output']).lower() for target in g['targets']]
            y_real.append(t)
            y_pred.append(y_predict[i].strip().lower())

         # Tokenize y_real and y_pred
        y_real_tokenized = [[nltk.word_tokenize(sent) for sent in ref] for ref in y_real]
        y_pred_tokenized = [nltk.word_tokenize(sent) for sent in y_pred]

        # Calculate BLEU score
        chencherry = SmoothingFunction()
        bleu_score = corpus_bleu(y_real_tokenized, y_pred_tokenized, smoothing_function=chencherry.method3)

        result += '\n' + 'Task: '+ "Lexicalization"
        result += '\n' + 'Set: '+ _set 
        result += '\n' + 'Model: '+ model 
        result += '\n' + 'Result: '+ str(round(bleu_score, 2))
        result += '\n' + 20 * '-' + '\n'
        # result += '\n' + 40 * '-' + '\n' 

write_file(write_dir, result, mode='a')
write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a') 


print('Seen domains')
result = '\nSeen domains:  \n' 

for _set in ['test', 'dev']:
    gold_path=os.path.join(path_input, 'lexicalization', _set + '.json') #path/task/set.json
    gold = read_file(gold_path)
    
    for model in ["t5", "bart", "gpt2"]:
        p = os.path.join(path_result, 'lexicalization', model, 'lexicalization'+'_'+_set +'.txt')
        # p=/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/reg/t5/reg_test.txt
        y_predict = [i.replace('[', '<').replace(']', '>') for i in read_file_map(p)]
               
        y_real, y_pred = [], []
        for i, g in enumerate(gold):
            if g['category'] not in unseen_domains:
                t = [' '.join(target['output']).lower() for target in g['targets']]
                y_real.append(t)
                y_pred.append(y_predict[i].strip().lower())

         # Tokenize y_real and y_pred
        y_real_tokenized = [[nltk.word_tokenize(sent) for sent in ref] for ref in y_real]
        y_pred_tokenized = [nltk.word_tokenize(sent) for sent in y_pred]

        # Calculate BLEU score
        chencherry = SmoothingFunction()
        bleu_score = corpus_bleu(y_real_tokenized, y_pred_tokenized, smoothing_function=chencherry.method3)

        result += '\n' + 'Task: '+ "Lexicalization"
        result += '\n' + 'Set: '+ _set 
        result += '\n' + 'Model: '+ model 
        result += '\n' + 'Result: '+ str(round(bleu_score, 2))
        result += '\n' + 20 * '-' + '\n'
        # result += '\n' + 40 * '-' + '\n' 

write_file(write_dir, result, mode='a')
write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')


print('Unseen domains')
result = '\nUnseen domains:  \n' 

for _set in ['test', 'dev']:
    gold_path=os.path.join(path_input, 'lexicalization', _set + '.json') #path/task/set.json
    gold = read_file(gold_path)
    
    for model in ["t5", "bart", "gpt2"]:
        p = os.path.join(path_result, 'lexicalization', model, 'lexicalization'+'_'+_set +'.txt')
        # p=/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/reg/t5/reg_test.txt
        y_predict = [i.replace('[', '<').replace(']', '>') for i in read_file_map(p)]
               
        y_real, y_pred = [], []
        for i, g in enumerate(gold):
            if g['category'] in unseen_domains:
                t = [' '.join(target['output']).lower() for target in g['targets']]
                y_real.append(t)
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
        result += '\n' + 'Result: '+ str(round(bleu_score, 2))
        result += '\n' + 20 * '-' + '\n'
        # result += '\n' + 40 * '-' + '\n' 

write_file(write_dir, result, mode='a')
write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a') 


################################################
    #  Evaluation of Final Texts (BLEU) #
################################################
print('Evaluation of Final Texts (BLEU)')
print('All domains')

result = '\n\nEvaluation of Final Texts (BLEU)\n'
result += '\nAll domains:\n'
write_file(write_dir, result, mode='a')

models = ["t5", "bart", "gpt2"]
tasks = ['end2end', 'sr']

for _set in ['test', 'dev']:
    gold_path = os.path.join(path_input, 'end2end', _set + '.json')  # path/task/set.json
    gold = read_file(gold_path)

    for kind in tasks:
        for model in models:
            if _set == 'dev':
                _set = 'eval'
            p = os.path.join(path_result, kind, model, kind + '_pipeline_' + _set + '.txt')
            y_predict = read_file_map(p)

            y_real, y_pred = [], []
            for i, g in enumerate(gold):
                t = [' '.join(target['output']).lower() for target in g['targets']]
                y_real.append(t)
                y_pred.append(y_predict[i].strip().lower())

            y_real_tokenized = [[nltk.word_tokenize(sent) for sent in ref] for ref in y_real]
            y_pred_tokenized = [nltk.word_tokenize(sent) for sent in y_pred]

            chencherry = SmoothingFunction()
            try:
                bleu_score = corpus_bleu(y_real_tokenized, y_pred_tokenized, smoothing_function=chencherry.method3)
            except:
                bleu_score = 0

            result = '\n' + 'Task: ' + kind
            result += '\n' + 'Set: ' + _set
            result += '\n' + 'Model: ' + model
            result += '\n' + 'Result: ' + str(round(bleu_score, 2))
            result += '\n' + 20 * '-' + '\n'

            write_file(write_dir, result, mode='a')

write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')



print('Seen domains')
result = '\nSeen domains:\n'
write_file(write_dir, result, mode='a')

models = ["t5", "bart", "gpt2"]
tasks = ['end2end', 'sr']

for _set in ['test', 'dev']:
    gold_path = os.path.join(path_input, 'end2end', _set + '.json')  # path/task/set.json
    gold = read_file(gold_path)

    for kind in tasks:
        for model in models:
            if _set == 'dev':
                _set = 'eval'
            p = os.path.join(path_result, kind, model, kind + '_pipeline_' + _set + '.txt')
            y_predict = read_file_map(p)

            y_real, y_pred = [], []
            for i, g in enumerate(gold):
                if g['category'] not in unseen_domains:
                    t = [' '.join(target['output']).lower() for target in g['targets']]
                    y_real.append(t)
                    y_pred.append(y_predict[i].strip().lower())

            y_real_tokenized = [[nltk.word_tokenize(sent) for sent in ref] for ref in y_real]
            y_pred_tokenized = [nltk.word_tokenize(sent) for sent in y_pred]

            chencherry = SmoothingFunction()
            try:
                bleu_score = corpus_bleu(y_real_tokenized, y_pred_tokenized, smoothing_function=chencherry.method3)
            except:
                bleu_score = 0 

            result = '\n' + 'Task: ' + kind
            result += '\n' + 'Set: ' + _set
            result += '\n' + 'Model: ' + model
            result += '\n' + 'Result: ' + str(round(bleu_score, 2))
            result += '\n' + 20 * '-' + '\n'

            write_file(write_dir, result, mode='a')

write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')


print('Unseen domains')
result = '\nUnseen domains:\n'
write_file(write_dir, result, mode='a')

models = ["t5", "bart", "gpt2"]
tasks = ['end2end', 'sr']

for _set in ['test', 'dev']:
    gold_path = os.path.join(path_input, 'end2end', _set + '.json')  # path/task/set.json
    gold = read_file(gold_path)

    for kind in tasks:  # Iterate over tasks
        for model in models:
            if _set == 'dev':
                _set = 'eval'
            p = os.path.join(path_result, kind, model, kind + '_pipeline_' + _set + '.txt')
            y_predict = read_file_map(p)

            y_real, y_pred = [], []
            for i, g in enumerate(gold):
                if g['category'] in unseen_domains:
                    t = [' '.join(target['output']).lower() for target in g['targets']]
                    y_real.append(t)
                    y_pred.append(y_predict[i].strip().lower())

            y_real_tokenized = [[nltk.word_tokenize(sent) for sent in ref] for ref in y_real]
            y_pred_tokenized = [nltk.word_tokenize(sent) for sent in y_pred]

            chencherry = SmoothingFunction()
            try:
                bleu_score = corpus_bleu(y_real_tokenized, y_pred_tokenized, smoothing_function=chencherry.method3)
            except:
                bleu_score = 0 

            result = '\n' + 'Task: ' + kind
            result += '\n' + 'Set: ' + _set
            result += '\n' + 'Model: ' + model
            result += '\n' + 'Result: ' + str(round(bleu_score, 2))
            result += '\n' + 20 * '-' + '\n'

            write_file(write_dir, result, mode='a')

write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')




################################################
    #  Evaluation of Final Texts (METEOR) #
################################################
print('Evaluation of Final Texts (METEOR)')
print('All domains')

result = '\n\nEvaluation of Final Texts (METEOR)\n'
result += '\nAll domains:\n'
write_file(write_dir, result, mode='a')

models = ["t5", "bart", "gpt2"]
tasks = ['end2end', 'sr']

for _set in ['test', 'dev']:
    gold_path = os.path.join(path_input, 'end2end', _set + '.json')  # path/task/set.json
    gold = read_file(gold_path)

    for kind in tasks:
        for model in models:
            if _set == 'dev':
                _set = 'eval'
            p = os.path.join(path_result, kind, model, kind + '_pipeline_' + _set + '.txt')
            y_predict = read_file_map(p)

            y_real, y_pred = [], []
            for i, g in enumerate(gold):
                targets = [nltk.word_tokenize(' '.join(target['output']).lower()) for target in g['targets']]
                y_real.extend(targets)
                pred = ' '.join(nltk.word_tokenize(y_predict[i].strip())).lower()
                y_pred.append(pred)


            # Calculate METEOR score
            try:
                y_real_str = '\n'.join([' '.join(sent) for sent in y_real])
                y_pred_str = '\n'.join([' '.join(sent) for sent in y_pred])
                meteor = meteor_score(y_real_str, y_pred_str)
            except ZeroDivisionError:
                meteor = 0.0

            result = '\n' + 'Task: ' + kind
            result += '\n' + 'Set: ' + _set
            result += '\n' + 'Model: ' + model
            result += '\n' + 'Result: ' + str(round(meteor, 2))
            result += '\n' + 20 * '-' + '\n'

            write_file(write_dir, result, mode='a')

write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')



print('Seen domains')
result = '\nSeen domains:\n'
write_file(write_dir, result, mode='a')

models = ["t5", "bart", "gpt2"]
tasks = ['end2end', 'sr']

for _set in ['test', 'dev']:
    gold_path = os.path.join(path_input, 'end2end', _set + '.json')  # path/task/set.json
    gold = read_file(gold_path)

    for kind in tasks:
        for model in models:
            if _set == 'dev':
                _set = 'eval'
            p = os.path.join(path_result, kind, model, kind + '_pipeline_' + _set + '.txt')
            y_predict = read_file_map(p)

            y_real, y_pred = [], []
            for i, g in enumerate(gold):
                if g['category'] not in unseen_domains:
                    targets = [nltk.word_tokenize(' '.join(target['output']).lower()) for target in g['targets']]
                    y_real.extend(targets)
                    pred = ' '.join(nltk.word_tokenize(y_predict[i].strip())).lower()
                    y_pred.append(pred)

            # Calculate METEOR score
            try:
                y_real_str = '\n'.join([' '.join(sent) for sent in y_real])
                y_pred_str = '\n'.join([' '.join(sent) for sent in y_pred])
                meteor = meteor_score(y_real_str, y_pred_str)
            except ZeroDivisionError:
                meteor = 0.0

            result = '\n' + 'Task: ' + kind
            result += '\n' + 'Set: ' + _set
            result += '\n' + 'Model: ' + model
            result += '\n' + 'Result: ' + str(round(meteor, 2))
            result += '\n' + 20 * '-' + '\n'

            write_file(write_dir, result, mode='a')

write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')



print('Unseen domains')
result = '\nUnseen domains:\n'
write_file(write_dir, result, mode='a')

models = ["t5", "bart", "gpt2"]
tasks = ['end2end', 'sr']

for _set in ['test', 'dev']:
    gold_path = os.path.join(path_input, 'end2end', _set + '.json')  # path/task/set.json
    gold = read_file(gold_path)

    for kind in tasks:
        for model in models:
            if _set == 'dev':
                _set = 'eval'
            p = os.path.join(path_result, kind, model, kind + '_pipeline_' + _set + '.txt')
            y_predict = read_file_map(p)

            y_real, y_pred = [], []
            for i, g in enumerate(gold):
                if g['category'] in unseen_domains:
                    targets = [nltk.word_tokenize(' '.join(target['output']).lower()) for target in g['targets']]
                    y_real.extend(targets)
                    pred = ' '.join(nltk.word_tokenize(y_predict[i].strip())).lower()
                    y_pred.append(pred)

            # Calculate METEOR score
            try:
                y_real_str = '\n'.join([' '.join(sent) for sent in y_real])
                y_pred_str = '\n'.join([' '.join(sent) for sent in y_pred])
                meteor = meteor_score(y_real_str, y_pred_str)
            except ZeroDivisionError:
                meteor = 0.0

            result = '\n' + 'Task: ' + kind
            result += '\n' + 'Set: ' + _set
            result += '\n' + 'Model: ' + model
            result += '\n' + 'Result: ' + str(round(meteor, 2))
            result += '\n' + 20 * '-' + '\n'

            write_file(write_dir, result, mode='a')

write_file(write_dir, '\n' + 60 * '#' + '\n', mode='a')
################################################
    #  Evaluation of Final Texts (Fluency and Semantic) #
################################################