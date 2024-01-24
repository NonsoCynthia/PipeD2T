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

    # device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(
            load_in_8bit=False,
            load_in_4bit=False,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype="float32",
       )

   # bnb_config = BitsAndBytesConfig(load_in_4bit=True, #quantization
                                   # bnb_4bit_quant_type='nf4',
                                    #bnb_4bit_use_double_quant=True,
                                    #bnb_4bit_compute_dtype=bfloat16,
                                    #)
    

    model_config = AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth,
        # load_in_4bit=False
    )

    model.eval()
    print(f"Model loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        use_auth_token=hf_auth
    )

    query_pipeline = pipeline(
        "text-generation",
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

    # test1 = test_model(tokenizer, query_pipeline, "Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words.")
    test1 = query_pipeline("Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words.") 
    # print('huggin face:', test1[0]["generated_text"])
  
    llm = HuggingFacePipeline(pipeline=query_pipeline, model_kwargs={'temperature':0})
    # prompt="Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words."
    # test2 = llm.invoke(prompt)
    # print('llm:', test2)

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

    # query it
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

    


    

    # print(f'dataset_dict: {type(dataset_dict)}') #dataset_dict: <class 'datasets.dataset_dict.DatasetDict'>
    # print(f'train_dataset: {type(train_dataset)}') #train_dataset: <class 'data.load_dataset.CustomDataset'>


    # Create 3 randomly selected FewShot examples
    # train_examples = "" 
    # for i in range(0, 5):  
    #     context = f'''\n Example {i+1}:\n  {train_dataset['Source'][i]}\n  Desired Output: {train_dataset['Target'][i]}'''
    #     train_examples += context
    # train_examples += "\n" 

    # t_task = {
    #     "ordering": "order",
    #     "structuring": "structure",
    #     "lexicalization": "lexicalized format",  
    #     "reg": "refering expression generation format",  
    #     "sr": "textual realization format"
    # }

#     eg = os.path.join(write_path, f'{task}_prompts.txt')
    
#     # Define examples outside the loop
#     examples = f'''I would like to arrange my triples in a specific {t_task[task]} to control the way information is expressed in the final summary. \
# Below, you'll find examples from my {task} dataset along with inputs and expected outputs:{train_examples}\
# Please provide the desired output for the next input. Print only the order: '''
    
    # write_file(eg, examples, mode='w')
    # with open(eg,'r') as f:
    #     examples = f.read()

    # print(examples)
    
    # # Import the validation datasets
    # evaluation = {
    #     f"{task}_dev": dataset_dict["validation"],
    #     f"{task}_test": dataset_dict["test"],
    #     f"{task}_pipeline_eval": dataset_dict["pipeline_eval"],
    #     f"{task}_pipeline_test": dataset_dict["pipeline_test"]
    # }

    # # # Feed the chatgpt the dev, test, and pipeline datasets for inference
    # for dataset_name, dataset in evaluation.items():
    #     print(f'{dataset_name}.txt')
    #     path = os.path.join(write_path, f'{dataset_name}.txt')
    #     feedback = []
    #     for item in dataset:
    #         prompt = f"{examples}{item['Source']}"
    #         response = get_completion(prompt, model) 
    #         feedback.append(response.replace('Desired Output: ',''))
    #         # print(response)
    #         # using sleep() to hault the code executions
    #         # time.sleep(30)
    #     write_file(path, '\n'.join(feedback), mode='a')  # Write your result into a file
    
    #     print(f'{dataset_name}.txt Ended!!!!', "\n")

# salloc --gres=gpu:rtx2080ti:1
# salloc --gres=gpu:rtxa6000:1
