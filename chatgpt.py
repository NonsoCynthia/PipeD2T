import os
import time
import json
import argparse
import random
from dotenv import load_dotenv, find_dotenv
from data.load_dataset import CustomDataset, preprocess_data, realize_date, read_file
from mapping import entity_mapping, prcs_entry, delist, split_triples
from openai import OpenAI
from aixplain.factories import ModelFactory

_ = load_dotenv(find_dotenv())  # read local .env file
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#def get_completion(prompt, model):  # model="gpt-3.5-turbo", 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo', 'gpt-4'
    #messages = [{"role": "user", "content": prompt}]
    #response = client.chat.completions.create(
        #model=model,
        #messages=messages,
        #temperature=0
    #)
    #return response.choices[0].message.content.strip()


TEAM_API_KEY=os.getenv("TEAM_API_KEY")
#model_gpt3 = ModelFactory.get("640b517694bf816d35a59125") #aixplain gpt-3.5

# Function to write files
def write_file(write_path, result, mode='w'):
    with open(write_path, mode) as f:
        f.write(result)

class Inferencer:
    def __init__(self, model, parameters, examples, task, dataset, dataset_name, write_path):
        self.model = model
        self.parameters = parameters
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.examples = examples
        self.write_path = write_path
        self.task = task

    def evaluate(self):
        path = os.path.join(self.write_path, f'{self.dataset_name}.txt')
        feedback = []
        for i, item in enumerate(self.dataset):
            prompt = f"{self.examples}\"\"\"{item['Source']}\"\"\" \nOutput:"
            if "gpt" in self.write_path:
                data = [{"role": "user", "content": prompt}]
                result = self.model.run(data, parameters=self.parameters)
                while 'data' not in result:
                    print(f"No 'data' key found in the result for dataset '{self.dataset_name}', item {i}. Retrying...")
                    result = self.model.run(data, parameters=self.parameters)
                if self.task == "reg":
                    result = result['data'].strip().replace('\n',' ').replace('[','').replace(']','').replace("#","")
                    result = result[:-1] if result.endswith('.') and len(result) > 1 else result
                else:
                    result = result['data'].strip().replace('\n',' ')                       
            else:
                result = self.model.run(prompt)
                while 'data' not in result:
                    print(f"No 'data' key found in the result for dataset '{self.dataset_name}', item {i}. Retrying...")
                    result = self.model.run(prompt)
                result = result['data'].replace('\n', ' ').strip()#.split("Output:")[-1].split('\n')[0].strip()            
            print(f"Input {i}: {result}")
            feedback.append(result)
        write_file(path, '\n'.join(feedback), mode='w')  # Write your result into a file
        print(f'{self.dataset_name}.txt Ended!!!!', "\n")

    def evaluate_reg(self):
        source = self.dataset['Source']
        targets = self.dataset['Target']
        source = [split_triples(t.split()) for t in source]
        entity_maps = [entity_mapping(t) for t in source]
        entity_maps = [dict((k, v.replace('_', ' ').replace('\"', ' ').replace('\'', ' ').strip()) for k, v in entity_mapping(t).items()) for t in source]
        feedback = []  # Rename the inner results list to results
        # Loop through the targets
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
                                if clean_token in entity_maps[y]:
                                    pos_context.append(entity_maps[y][clean_token])
                                else:
                                    pos_context.append(clean_token.lower())
                            go_in = (delist(pre_context) + ' ' + delist(pos_context) + '. ' + entity).strip()
                            print(f"go_in-{i}: {go_in}")
                            prompt = f"{self.examples}\"\"\"{go_in}\"\"\" \nOutput:"
                            if "gpt" in self.write_path:
                                data = [{"role": "user", "content": prompt}]
                                #Run the inference
                                result = self.model.run(data, parameters=self.parameters)
                                while 'data' not in result:
                                    print(f"No 'data' key found in the result for dataset '{self.dataset_name}', item {i}. Retrying...")
                                    result = self.model.run(data, parameters=self.parameters)
                                #Extract the Inference
                                result = result['data'].strip().replace('\n', ' ').replace('[','').replace(']','').replace("#","")
                                result = result[:-1] if result.endswith('.') and len(result) > 1 else result
                            else:
                                #Run the inference
                                result = self.model.run(prompt)
                                while 'data' not in result:
                                    print(f"No 'data' key found in the result for dataset '{self.dataset_name}', item {i}. Retrying...")
                                    result = self.model.run(prompt)
                                #Extract the Inference
                                result = result['data'].replace('\n',' ').strip() #.split("Output:")[-1].split('\n')[0].strip()
                                result = result[:-1] if result.endswith('.') and len(result) > 1 else result

                            #assign the Inference to the REG position in the text
                            refex = result.strip()
                    entry[i] = refex
                    pre_context.append(entity)
                else: 
                    pre_context.append(token.lower())
            ent = ' '.join(entry).replace('  .', '.').replace(' .', '.').replace(' ,', ',').replace(' !', '!') + '.'
            #print(f"Input {y}: {ent}")
            feedback.append(ent)

        path = os.path.join(self.write_path, f'{self.dataset_name}.txt')
        write_file(path, '\n'.join(feedback), mode='w')  # Write your result into a file
        print(f'{self.dataset_name}.txt Ended!!!!', "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", help="model ID in aiXplain")
    parser.add_argument("--model_path", help="path to the model")
    parser.add_argument("--task", help="Training task")
    parser.add_argument("--data_path", help="path to the data")
    parser.add_argument("--write_path", help="path to write best model")
    args = parser.parse_args()

    # Model settings, Settings and configurations
    model_id = args.model_id
    task = args.task
    data = args.data_path
    model_path = args.model_path #"gpt-3.5"
    write_path = os.path.join(f"{args.write_path}", f"{args.task}", f"{args.model_path}")

    # Create result directory if it doesn't exist.
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    prompt_dir = os.path.join(f"{args.write_path}", f"{args.task}")
    eg = os.path.join(prompt_dir, f'{task}_prompts.txt')

    t_task = {
        "ordering": "order",
        "structuring": "structure",
        "lexicalization": "lexicalized format",
        "reg": "refering expression generation format",
        "sr": "textual realization format",
        "end2end": "Summary"
        }

    # print('Read dataset')
    dataset_dict = preprocess_data(data, task, model_path)
    train_dataset = CustomDataset(dataset_dict["train"])

    # Create 3 randomly selected FewShot examples
    indices = list(range(len(train_dataset['Source'])))
    random.shuffle(indices)
    # Generate a random starting index
    start_index = random.randint(0, len(indices) - 5)
    train_examples = ""
    for i in range(start_index, start_index + 5):
        index = indices[i]
        context = f'''\n Example {i - start_index + 1}: """\n{train_dataset['Source'][index]}\n  """\n  Output: {train_dataset['Target'][index]}\n ###'''
        train_examples += context
    train_examples += "\n"
  
    # Define examples outside the loop
    #examples = f'''I would like to arrange my triples in a specific {t_task[task]} to control the way information is expressed in the final summary. Below, you'll find examples from my {task} dataset along with inputs and expected outputs:{train_examples}Now strictly generate all the output result for the query, extra comments is not allowed. Do not dismiss numbers in digits.\nQuery: ''' 

    #examples_e2e = f'''I would like you to generate summaries from the triples provided. Below you'll find examples of the input triples and the expected summary outputs. {train_examples}Now strictly generate the summaries for the query, extra comments is not allowed. Do not dismiss numbers in digits. \nQuery:'''
    

    examples = read_file(eg)

    #Write the prompt into a txt file
    #write_file(eg, examples_e2e, mode='w')

    print(examples)
    
    ## Import the validation datasets
    evaluation = {
        f"{task}_dev": dataset_dict["validation"],
        f"{task}_test": dataset_dict["test"],
        f"{task}_pipeline_eval": dataset_dict["pipeline_eval"],
        f"{task}_pipeline_test": dataset_dict["pipeline_test"]
    }

    model = ModelFactory.get(f"{model_id}") #gpt-3.5
    parameters = {"temperature": 0, "max_tokens": 1024, "top_p": 1}

    for dataset_name, dataset in evaluation.items():
        inf = Inferencer(model, parameters, examples, task, dataset, dataset_name, write_path)
        print(f'Evaluating {dataset_name}.txt')
        if task == "reg" and dataset_name not in [f"{task}_dev", f"{task}_test"]:
            inf.evaluate_reg()
        else:
            inf.evaluate()
