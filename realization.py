import os
import sys
import json
import operator
import argparse
from data.load_dataset import CustomDataset, preprocess_data

SURFACE_PATH='/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/data/deepnlg/input/sr/surfacevocab.json'

class Realization():
    def __init__(self, rule_path):
        self.surface_rules = json.load(open(rule_path))

    def realize(self, entry):
        template = entry
        stemplate = entry.split()

        for i, token in enumerate(stemplate):
            if 'VP[' == token[:3]:
                try:
                    rule = token.strip() + ' ' + stemplate[i+1].strip()
                    try:
                        surface_rule = max(self.surface_rules[rule].items(), key=operator.itemgetter(1))[0]
                        template = template.replace(rule, surface_rule)
                    except:
                        template = template.replace(rule, stemplate[i+1].strip())
                except:
                    template = template.replace(token.strip(), ' ')
            elif 'DT[' == token[:3]:
                rule = token.strip() + ' ' + stemplate[i+1].strip()
                if token.strip() == 'DT[form=undefined]':
                    if stemplate[i+2].strip().lower()[0] in ['a', 'e', 'i', 'o', 'u']:
                        template = template.replace(rule, 'an')
                    else:
                        template = template.replace(rule, 'a')
                else:
                    try:
                        surface_rule = max(self.surface_rules[rule].items(), key=operator.itemgetter(1))[0]
                        template = template.replace(rule, surface_rule)
                    except:
                        template = template.replace(rule, stemplate[i+1].strip())
        template = template.replace('-LRB-', '(').replace('-RRB-', ')')
        return template

        
    def __call__(self, in_path, out_path):
        result = [self.realize(entry) for entry in in_path]
        with open(out_path, 'w') as f:
            f.write('\n'.join(result))



class Realization_():
    def __init__(self, rule_path):
        self.surface_rules = json.load(open(rule_path))
    
    def realize(self, entry):
        template = entry
        stemplate = entry.split()

        i = 0
        while i < len(stemplate):
            token = stemplate[i]
            if 'VP[' == token[:3] and i + 1 < len(stemplate):
                rule = token.strip() + ' ' + stemplate[i + 1].strip()
                try:
                    surface_rule = max(self.surface_rules.get(rule, {}).items(), key=operator.itemgetter(1))[0]
                    template = template.replace(rule, surface_rule)
                    i += 1  # Incrementing index when successfully processed
                except (ValueError, IndexError):
                    template = template.replace(rule, stemplate[i + 1].strip())
                    i += 1  # Incrementing index when successfully processed
            elif 'DT[' == token[:3] and i + 1 < len(stemplate):
                rule = token.strip() + ' ' + stemplate[i + 1].strip()
                try:
                    surface_rule = max(self.surface_rules.get(rule, {}).items(), key=operator.itemgetter(1))[0]
                    template = template.replace(rule, surface_rule)
                    i += 2  # Incrementing index when successfully processed
                except (ValueError, IndexError):
                    template = template.replace(rule, stemplate[i + 1].strip())
                    i += 1  # Incrementing index when successfully processed
            else:
                i += 1  # Increment index for other cases

        template = template.replace('-LRB-', '(').replace('-RRB-', ')')
        return template

    def __call__(self, in_path, out_path):
        result = [self.realize(entry) for entry in in_path]
        with open(out_path, 'w') as f:
            f.write('\n'.join(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to the model")
    parser.add_argument("--task", help="Training task")
    parser.add_argument("--surface_path", help="path to the surface vocab") 
    parser.add_argument("--write_path", help="path to write best model")
    # parser.add_argument("--data_path", help="path to the data")
    args = parser.parse_args()

    # Model settings, Settings and configurations
    model = args.model
    task = args.task
    SURFACE_PATH = args.surface_path
    write_path = os.path.join(args.write_path, task, model) #/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/sr/t5
    # data = args.data_path # Input data path. This should be the path to the previous pipeline task, in this case it should be reg

    # Create result directory if it doesn't exist.
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    dataset_dict = preprocess_data(write_path, task, model)
    evaluation = {
        f"{task}_pipeline_eval": dataset_dict["pipeline_eval"],
        f"{task}_pipeline_test": dataset_dict["pipeline_test"]
    }

    if model == "bart":
        sr_model = Realization_(SURFACE_PATH)
    else:
        sr_model = Realization(SURFACE_PATH)
    
    for dataset_name, dataset in evaluation.items():
        print(f'{dataset_name}.txt')
        path = os.path.join(write_path, f'{dataset_name}.txt')
        sr_model(in_path=dataset['Source'], out_path=path)