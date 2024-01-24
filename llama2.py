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
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline

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
        
def generate_text(model_id, hf_auth, input_text, max_length, device="cuda"):
    # Load model configuration
    model_config = AutoConfig.from_pretrained(model_id, use_auth_token=hf_auth)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                config=model_config,
                                                trust_remote_code=True,
                                                use_auth_token=hf_auth,
                                                load_in_4bit=True,
                                                )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate output using the model
    generated_ids = model.generate(input_ids, max_length=max_length, num_beams=2, length_penalty=1.0)

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

    # print('Read dataset')
    dataset_dict = preprocess_data(data, task, model_path)
    train_dataset = dataset_dict["train"]

    model_id = "meta-llama/Llama-2-70b-chat-hf" #"meta-llama/Llama-2-13b-chat-hf" #"meta-llama/Llama-2-7b-chat-hf"
    hf_auth = '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'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_text = "Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words."
    max_length = 300
    generated_text = generate_text(model_id, hf_auth, input_text, max_length)
    print("Generated Text:", generated_text)