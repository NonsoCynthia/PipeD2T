import os
import sys
import re
import json
import operator
import argparse
from data.load_dataset import CustomDataset, preprocess_data

SURFACE_PATH='/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/data/deepnlg/input/sr/surfacevocab.json'

def clean_and_summarize_text(xcontents):
    # Add ". " before any occurrence of "Is there anything" if there is no ". " preceding it
    if re.search(r"Is there anything\s*\b[^.]*$", xcontents) and ". " not in xcontents:
        match = re.search(r"Is there anything\s*\b[^.]*$", xcontents)
        if match:
            index = match.start()
            if index > 0 and xcontents[index - 1] != ".":
                xcontents = xcontents[:index] + ". " + xcontents[index:]
    
    if ": " in xcontents:
        xcontents = xcontents.split(": ")[-1]
    sentences = xcontents.split(". ")
    for i in reversed(range(len(sentences))):
        if sentences[i].endswith(("?", "!", "!.", "?.")):
            sentences.pop(i)
    if sentences and sentences[0].startswith("]"):
        sentences[0] = sentences[0].replace("]", "").strip()  # Corrected line
    summary = ". ".join(sentences)
    return summary.strip()

def process_txt_file(content):
    try:      
        result = []
        for text in content:
            text = text.replace("..", ".").replace("VP [", "VP[")
            processed_content = clean_and_summarize_text(text)
            processed_content = (processed_content+".").replace("..", ".")
            result.append(processed_content)
        return result
    except Exception as e:
        print("Error:", e)
        return None


def clean_text(text):
    # Perform text cleaning operations here
    cleaned_text = text.strip()  # Remove leading and trailing whitespace
    cleaned_text = cleaned_text.lstrip()  # Remove trailing whitespaces at the beginning of the text
    cleaned_text = cleaned_text.capitalize()  # Make the first letter uppercase
    cleaned_text = cleaned_text.replace(" '", "'").replace(" .", ".").replace(" ,", ",")
    cleaned_text = cleaned_text.replace(" be ", " is ")
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with single space
    cleaned_text = re.sub(r'(\b[A-Za-z]+)_([A-Za-z]+\b)', r'\1 \2', cleaned_text)  # Replace underscores with spaces
    return cleaned_text


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
                if i + 2 < len(stemplate):  # Check if there are enough tokens left in the list
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
        result = [clean_text(entry) for entry in result]  # Clean each entry
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
        result = [clean_text(entry) for entry in result]  # Clean each entry
        with open(out_path, 'w') as f:
            f.write('\n'.join(result))

class Realization_Prompt():
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
        result = [entry for entry in in_path]
        result = process_txt_file(result)
        result = [self.realize(entry) for entry in result]
        with open(out_path, 'w') as f:
            f.write('\n'.join(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to the model")
    parser.add_argument("--task", help="Training task")
    parser.add_argument("--surface_path", help="path to the surface vocab") 
    parser.add_argument("--write_path", help="path to write best model")
    parser.add_argument("--data_path", help="path to the data")
    args = parser.parse_args()

    # Model settings, Settings and configurations
    model = args.model
    task = args.task
    SURFACE_PATH = args.surface_path
    write_path = os.path.join(args.write_path, f"{task}/{model}") #/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/sr/t5
    data_path = args.data_path # Input data path. This should be the path to the previous pipeline task, in this case it should be reg

    # Create result directory if it doesn't exist.
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    dataset_dict = preprocess_data(data_path, task, model)
    evaluation = {
        f"{task}_pipeline_eval": dataset_dict["pipeline_eval"],
        f"{task}_pipeline_test": dataset_dict["pipeline_test"]
    }

    if model == "bart":
        sr_model = Realization_(SURFACE_PATH)
    elif model == "cohere":
        sr_model = Realization_Prompt(SURFACE_PATH)
    else:
        sr_model = Realization(SURFACE_PATH)
    
    for dataset_name, dataset in evaluation.items():
        print(f'Surface Realization {dataset_name}.txt for {model}')
        path = os.path.join(write_path, f'{dataset_name}.txt')
        sr_model(in_path=dataset['Source'], out_path=path)
