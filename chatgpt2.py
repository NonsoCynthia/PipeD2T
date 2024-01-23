import os
import time
import argparse
from dotenv import load_dotenv, find_dotenv
from data.load_dataset import CustomDataset, preprocess_data
from openai import OpenAI

_ = load_dotenv(find_dotenv())  # read local .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_completion(prompt, model):  # model="gpt-3.5-turbo", 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo', 'gpt-4'
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Function to write files
def write_file(write_path, result, mode='w'):
    with open(write_path, mode) as f:
        f.write(result)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", help="path to the model")
    # parser.add_argument("--task", help="Training task")
    # parser.add_argument("--data_path", help="path to the data")
    # parser.add_argument("--write_path", help="path to write best model")
    # args = parser.parse_args()

    # # Model settings, Settings and configurations
    # model = args.model
    # task = args.task
    # data = args.data_path
    # write_path = os.path.join(args.write_path, task+"/chatgpt")

    model = "gpt-3.5-turbo"
    model_path = "chatgpt"
    task = "lexicalization" #structuring
    data = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/data/deepnlg"
    write_path = f"/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/{task}/chatgpt"

    # Create result directory if it doesn't exist.
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    # print('Read dataset')
    dataset_dict = preprocess_data(data, task, model_path)
    train_dataset = CustomDataset(dataset_dict["train"])

    # Create 3 randomly selected FewShot examples
    train_examples = "" 
    for i in range(0, 5):  
        context = f'''\n Example {i+1}:\n  {train_dataset['Source'][i]}\n  Desired Output: {train_dataset['Target'][i]}'''
        train_examples += context
    train_examples += "\n" 

    t_task = {
        "ordering": "order",
        "structuring": "structure",
        "lexicalization": "lexicalized format",  
        "reg": "refering expression generation format",  
        "sr": "textual realization format"
    }

    eg = os.path.join(write_path, f'{task}_prompts.txt')
    
    # Define examples outside the loop
    examples = f'''I would like to arrange my triples in a specific {t_task[task]} to control the way information is expressed in the final summary. \
Below, you'll find examples from my {task} dataset along with inputs and expected outputs:{train_examples}\
Please provide the desired output for the next input. Print only the order: '''
    
    write_file(eg, examples, mode='w')
    # with open(eg,'r') as f:
    #     examples = f.read()

    print(examples)
    
    # # Import the validation datasets
    evaluation = {
        f"{task}_dev": dataset_dict["validation"],
        f"{task}_test": dataset_dict["test"],
        f"{task}_pipeline_eval": dataset_dict["pipeline_eval"],
        f"{task}_pipeline_test": dataset_dict["pipeline_test"]
    }

    # # Feed the chatgpt the dev, test, and pipeline datasets for inference
    for dataset_name, dataset in evaluation.items():
        print(f'{dataset_name}.txt')
        path = os.path.join(write_path, f'{dataset_name}.txt')
        feedback = []
        for item in dataset:
            prompt = f"{examples}{item['Source']}"
            response = get_completion(prompt, model) 
            feedback.append(response.replace('Desired Output: ',''))
            # print(response)
            # using sleep() to hault the code executions
            # time.sleep(30)
        write_file(path, '\n'.join(feedback), mode='a')  # Write your result into a file
    
        print(f'{dataset_name}.txt Ended!!!!', "\n")

# salloc --gres=gpu:rtx2080ti:1
# salloc --gres=gpu:rtxa6000:1