import os
import time
import argparse
from dotenv import load_dotenv, find_dotenv
from data.load_dataset import CustomDataset, preprocess_data
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

    # model = "gpt-3.5-turbo"
    # model_path = "chatgpt"
    # task = "lexicalization" #structuring
    # data = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/data/deepnlg"
    # write_path = f"/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/{task}/chatgpt"

    # Create result directory if it doesn't exist.
    train_examples = ""
    for i in range(0, 5):
        context = f'''\n Example {i+1}:\n  {train_dataset['Source'][i]}\n  Output: {train_dataset['Target'][i]}'''
        train_examples += context
    train_examples += "\n"

    t_task = {
        "ordering": "order",
        "structuring": "structure",
        "lexicalization": "lexicalized format",
        "reg": "refering expression generation format",
        "sr": "textual realization format"
    }
    prompt_dir = os.path.join(f"{args.write_path}", f"{args.task}")
    eg = os.path.join(prompt_dir, f'{task}_prompts.txt')

    # Define examples outside the loop
    examples = f'''I would like to arrange my triples in a specific {t_task[task]} to control the way information is expressed in the final summary. \
Below, you'll find examples from my {task} dataset along with inputs and expected outputs:{train_examples}
Now strictly generate all the output result for the query, extra comments is not allowed.
Query: '''

    write_file(eg, examples, mode='w')

    print(examples)

    # # Import the validation datasets
    evaluation = {
        f"{task}_dev": dataset_dict["validation"],
        f"{task}_test": dataset_dict["test"],
        f"{task}_pipeline_eval": dataset_dict["pipeline_eval"],
        f"{task}_pipeline_test": dataset_dict["pipeline_test"]
    }

    model = ModelFactory.get(f"{model_id}") #gpt-3.5

    # # Feed the chatgpt the dev, test, and pipeline datasets for inference
    for dataset_name, dataset in evaluation.items():
        print(f'{dataset_name}.txt')
        path = os.path.join(write_path, f'{dataset_name}.txt')
        feedback = []
        for i, item in enumerate(dataset):
            prompt = f"{examples}{item['Source']} \nOutput:"
            result = model.run(prompt)
            output = result['rawData']['choices'][0]['message']['content']
            feedback.append(output.strip())

        write_file(path, '\n'.join(feedback), mode='a')  # Write your result into a file
        print(f'{dataset_name}.txt Ended!!!!', "\n")
