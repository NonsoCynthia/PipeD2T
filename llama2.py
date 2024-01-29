import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import time
import argparse
from dotenv import load_dotenv, find_dotenv
from data.load_dataset import preprocess_data
from openai import OpenAI
import pandas as pd
import torch
from torch import cuda, bfloat16
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline, AutoConfig

#from chromadb.config import Settings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings


# Function to write files
def write_file(write_path, result, mode='w'): 
    with open(write_path, mode) as f: 
        f.write(result) 
 
def get_prompt(instruction, new_system_prompt): 
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS 
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST 
    return prompt_template 
         
def generate_text(model, tokenizer, input_text, max_length, device="cuda"): 
 
    # Tokenize input text 
    input_ids = tokenizer.encode(input_text, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device).to(torch.long) 
 
     # Generate output using the model 
    with torch.no_grad(): 
        generated_ids = model.generate(input_ids.to(torch.long), do_sample=True, max_new_tokens=512, num_beams=2, length_penalty=1.0, temperature=1.0) 
 
    # Decode the generated output 
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True) 
 
    return generated_text.strip() 
 
 
if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", help="path to the model") 
    parser.add_argument("--task", help="Training task") 
    parser.add_argument("--data_path", help="path to the data") 
    parser.add_argument("--write_path", help="path to write best model") 
    args = parser.parse_args() 
 
    # # Model settings, Settings and configurations 
    model = args.model 
    task = args.task 
    data = args.data_path
    model_path = "llama2"
    write_path = os.path.join(args.write_path, task, model_path)
    # Create result directory if it doesn't exist.
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    dataset_dict = preprocess_data(data, task, model_path)
    train_dataset = dataset_dict["train"]

    model_id = "meta-llama/Llama-2-70b-chat-hf" #"meta-llama/Llama-2-13b-chat-hf" #"meta-llama/Llama-2-7b-chat-hf"
    hf_auth = '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'
    max_length = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # Load model configuration
    model_config = AutoConfig.from_pretrained(model_id, token=hf_auth)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                config=model_config,
                                                trust_remote_code=True,
                                                token=hf_auth,
                                                load_in_4bit=True,
                                                )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_auth) 
 
    # Add a padding token to the tokenizer 
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
 
    #input_text = "Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words." 
    #generated_text = generate_text(model, tokenizer, input_text, max_length) 
    #print("Generated Text:", generated_text) 

    # # Create 3 randomly selected FewShot examples
    # train_examples = "" 
    # for i in range(0, 5):  
    #     context = f'''\n Example {i+1}:\n  {train_dataset['Source'][i]}\n  Desired Output: {train_dataset['Target'][i]}'''
    #     train_examples += context
    # train_examples += "\n"

    # eg = os.path.join(write_path, f'{task}_prompts.txt')
 
    t_task = { 
        "ordering": "order", 
        "structuring": "structure", 
        "lexicalization": "lexicalized format", 
        "reg": "refering expression generation format", 
        "sr": "textual realization format" 
    } 
 
    B_INST, E_INST = "[INST]", "[/INST]" 
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n" 
    sys_prompt = """You are a professional computational linguist, and you are working on a traditional five module pipeline data-to-text generation: ordering->structuring->lexicalization->referring expression generation->surface realization. You will be given examples of triple texts and the desired outputs.""" 
 
 
    query = '''[TRIPLE] Italy capital Rome [/TRIPLE] [TRIPLE] A.S._Gubbio_1910 ground Italy [/TRIPLE] [TRIPLE] Italy language Italian_language [/TRIPLE] [TRIPLE] Italy leader Sergio_Mattarella [/TRIPLE]''' 
 
    train_examples = '''
    Example 1:
    [TRIPLE] Beef_kway_teow country "Singapore_and_Indonesia" [/TRIPLE] [TRIPLE] Beef_kway_teow ingredient Palm_sugar [/TRIPLE] [TRIPLE] Beef_kway_teow region Singapore [/TRIPLE] 
    Output: ingredient country 
    Example 2: 
    [TRIPLE] Bacon_Explosion country United_States [/TRIPLE] [TRIPLE] Bacon_Explosion ingredient Bacon [/TRIPLE] [TRIPLE] Bacon_Explosion mainIngredients "Bacon,sausage" [/TRIPLE] [TRIPLE] Bacon_Explosion region Kansas_City_metropolitan_area [/TRIPLE] 
    Output: mainIngredients region country 
    Example 3: 
    [TRIPLE] Adare_Manor architect "James_Pain_and_George_Richard_Pain," [/TRIPLE] [TRIPLE] Adare_Manor completionDate 1862 [/TRIPLE] [TRIPLE] Adare_Manor owner J._P._McManus [/TRIPLE] 
    Output: architect completionDate owner''' 

    examples = f'''I would like to arrange my triples in a specific {t_task[task]} to control the way information is expressed in the textual summary. \
Below, you'll find few examples from my {task} dataset along with inputs and expected outputs:{train_examples}. Please provide the desired output for the next input. Strictly print only the output of this input: '''

    # #prompt = get_prompt(instruction)
    # prompt = sys_prompt + instruction
    # prompt_gen = generate_text(model, tokenizer, prompt, max_length)
    # print(prompt_gen)

    # write_file(eg, examples, mode='w')
    # print(examples)

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
            response = generate_text(model, tokenizer, prompt, max_length) 
            feedback.append(response.split('Output:')[-1]) 
             
        write_file(path, '\n'.join(feedback), mode='a')  # Write your result into a file 
     
        print(f'{dataset_name}.txt Ended!!!!', "\n")
    
