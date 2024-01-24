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
from accelerate.utils import BnbQuantizationConfig


# Function to write files
def write_file(write_path, result, mode='w'):
    with open(write_path, mode) as f:
        f.write(result)

# def test_model(tokenizer, pipeline, prompt_to_test):
#     # adapted from https://huggingface.co/blog/llama2#using-transformers
#     sequences = pipeline(
#         prompt_to_test,
#         do_sample=True,
#         top_k=10,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         )
#     for seq in sequences:
#         print(f"Result: {seq['generated_text']}")
        
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

    # model = "gpt-3.5-turbo"
    # model_path = "llama2"
    # task = "ordering" #structuring
    # data = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/data/deepnlg"
    # write_path = f"/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/{task}/{model_path}"

    # Create result directory if it doesn't exist.
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    # print('Read dataset')
    dataset_dict = preprocess_data(data, task, model_path)
    train_dataset = dataset_dict["train"]

    model_id = "meta-llama/Llama-2-70b-chat-hf" #"meta-llama/Llama-2-13b-chat-hf" #"meta-llama/Llama-2-7b-chat-hf"
    hf_auth = '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'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

    model_config = transformers.AutoConfig.from_pretrained(model_id, use_auth_token=hf_auth)

    model = transformers.AutoModelForCausalLM.from_pretrained(model_id,
                                                            trust_remote_code=True,
                                                            config=model_config,
                                                            device_map='auto',
                                                            use_auth_token=hf_auth,
                                                            load_in_4bit=True
                                                        )

    model.eval()

    query_pipeline = pipeline("text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            max_new_tokens=512,
                            do_sample=True,
                            top_k=30,
                            num_return_sequences=1,
                            eos_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.1
                            )

    test1 = query_pipeline("Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words.") 
    print('hugging face:', test1[0]["generated_text"])
  
    llm = HuggingFacePipeline(pipeline=query_pipeline, model_kwargs={'temperature':0})
    prompt="Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words."
    test2 = llm.invoke(prompt)
    print('llm:', test2)

    datset = []
    for i in range(len(train_dataset)):
        text_entry = f"Source: {train_dataset['Source'][i]}; Target: {train_dataset['Target'][i]}"
        datset.append({"Text": text_entry})

    datset = pd.DataFrame(datset)

    loader = DataFrameLoader(datset, page_content_column="Text")
    # loader = DataFrameLoader(train_dataset, page_content_column="Source")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    train = text_splitter.split_documents(loader.load())
    # print(train[0])

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    
    vectordb = Chroma.from_documents(documents=train, embedding=embeddings, persist_directory="chroma_db")

    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectordb.as_retriever(), 
        verbose=True
    )

    # query = '''[TRIPLE] Alfred_Garth_Jones birthPlace England [/TRIPLE] [TRIPLE] Alfred_Garth_Jones deathPlace London [/TRIPLE] [TRIPLE] Alfred_Garth_Jones nationality United_Kingdom [/TRIPLE]'''
    query = '''[TRIPLE] Italy capital Rome [/TRIPLE] [TRIPLE] A.S._Gubbio_1910 ground Italy [/TRIPLE] [TRIPLE] Italy language Italian_language [/TRIPLE] [TRIPLE] Italy leader Sergio_Mattarella [/TRIPLE]'''
    # query = '''You are a professional computational linguist, and you are working on a data-to-text pipeline architecture 'ordering' to be specific.
    # This is an example of a triple set - [TRIPLE] subject predicate object [/TRIPLE]. I want you to output an ordering of these triplesets using the predicates only:''' 

    prompt = f'''I would like to arrange my triples in a specific order to control the way information is expressed in the final summary. Below, you'll find examples from my ordering dataset along with inputs and expected outputs:
    Example 1:
    [TRIPLE] Beef_kway_teow country "Singapore_and_Indonesia" [/TRIPLE] [TRIPLE] Beef_kway_teow ingredient Palm_sugar [/TRIPLE] [TRIPLE] Beef_kway_teow region Singapore [/TRIPLE]
    Desired output: ingredient country
    Example 2:
    [TRIPLE] Bacon_Explosion country United_States [/TRIPLE] [TRIPLE] Bacon_Explosion ingredient Bacon [/TRIPLE] [TRIPLE] Bacon_Explosion mainIngredients "Bacon,sausage" [/TRIPLE] [TRIPLE] Bacon_Explosion region Kansas_City_metropolitan_area [/TRIPLE]
    Desired output: mainIngredients region country
    Example 3:
    [TRIPLE] Adare_Manor architect "James_Pain_and_George_Richard_Pain," [/TRIPLE] [TRIPLE] Adare_Manor completionDate 1862 [/TRIPLE] [TRIPLE] Adare_Manor owner J._P._McManus [/TRIPLE]
    Desired output: architect completionDate owner
    Example 4:
    [TRIPLE] Ajoblanco country Spain [/TRIPLE] [TRIPLE] Ajoblanco ingredient Garlic [/TRIPLE] [TRIPLE] Ajoblanco mainIngredients "Bread,_almonds,_garlic,_water,_olive_oil" [/TRIPLE] [TRIPLE] Ajoblanco region Andalusia [/TRIPLE]
    Desired output: region country mainIngredients
    Example 5:
    [TRIPLE] Asilomar_Conference_Grounds added_to_the_National_Register_of_Historic_Places "1987-02-27" [/TRIPLE] [TRIPLE] Asilomar_Conference_Grounds architecture "Arts_and_Crafts_Movement_and_American_craftsman_Bungalows" [/TRIPLE] [TRIPLE] Asilomar_Conference_Grounds location "Asilomar_Blvd.,_Pacific_Grove,_California" [/TRIPLE] [TRIPLE] Asilomar_Conference_Grounds yearOfConstruction 1913 [/TRIPLE]
    Desired output: location yearOfConstruction added_to_the_National_Register_of_Historic_Places
    Please provide the desired output for this next input: {query}'''

    print(rag_pipeline.invoke(prompt)['result'])


   # bnb_config = BitsAndBytesConfig(load_in_4bit=True, #quantization
                                   # bnb_4bit_quant_type='nf4',
                                    #bnb_4bit_use_double_quant=True,
                                    #bnb_4bit_compute_dtype=bfloat16,
                                    #)