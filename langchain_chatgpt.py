# from openai import OpenAI
import os
import argparse
from dotenv import load_dotenv, find_dotenv
from data.load_dataset import CustomDataset, preprocess_data
from openai import OpenAI
# Import necessary libraries for Langchain
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.chains import LLMChain


_ = load_dotenv(find_dotenv())  # read local .env file
client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to the model")
    parser.add_argument("--task", help="Training task")
    parser.add_argument("--data_path", help="path to the data")
    parser.add_argument("--write_path", help="path to write best model")
    args = parser.parse_args()

    # Model settings, Settings and configurations
    model = args.model
    task = args.task
    data = args.data_path
    write_path = args.write_path + "/chatgpt"

    # Create result directory if it doesn't exist.
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    dataset_dict = preprocess_data(data, task, model)

    train_dataset = CustomDataset(dataset_dict["train"])

    # Create 10 randomly selected FewShot examples
    examples = []
    for i in range(0, 3):
        context = {"Input": f'''{train_dataset['Source'][i]}''', "Output": f'''{train_dataset['Target'][i]}'''}
        examples.append(context)

    ######OpenAI and LangChain ######
    # Create instances of Langchain objects
 




    langchain = OpenAI(model_name=model,
                api_key=os.getenv("OPENAI_API_KEY")
                )
    
    # create a example template
    example_template = """User: {Input}
AI: {Output} """

    example_prompt = PromptTemplate(
                                input_variables=["Input", "Output"], 
                                template=example_template
                                )

    # Feed examples and formatter to FewShotPromptTemplate. Finally, create a FewShotPromptTemplate object.
    # the prefix is our instructions
    prefix = f''''Below is a snippet of ten Input and Output entries in my {task} dataset. Provide answers to the subsequent inputs examples: '''
    # and the suffix our user input and output indicator
    suffix = """User: {Input}
AI: """
    prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix = prefix,
        suffix = suffix,
        input_variables=["Input"],
    )
    # testing = prompt_template.format(Input="[TRIPLE] Akeem_Priestley club RoPS [/TRIPLE] [TRIPLE] RoPS league Veikkausliiga [/TRIPLE]")
    # print(testing)
    # print(langchain(testing))

    evaluation = {
        f"{task}_dev": dataset_dict["validation"],
        f"{task}_test": dataset_dict["test"],
        # f"{task}_pipeline_eval": dataset_dict["pipeline_eval"],
        # f"{task}_pipeline_test": dataset_dict["pipeline_test"],
    }

    


    # # Feed the chatgpt the dev, test and pipeline datasets for inference
    # for dataset_name, dataset in evaluation.items():
    #     path = os.path.join(write_path, f'{dataset_name}.txt')
    #     feedback = []
    #     for item in dataset:
    #         prompt = prompt_template.format(Input=item['Source'])
    #         response = get_completion(prompt, model)
    #         print(response)
    #         feedback.append(response)

    #     write_file(path, '\n'.join(feedback), mode='w')  # Write your result into a file
            

