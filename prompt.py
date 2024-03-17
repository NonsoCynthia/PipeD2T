import os
import re
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader
from data.load_dataset import CustomDataset, preprocess_data, realize_date, read_file
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
hf_token = os.getenv("HF_TOKEN")

class LlamaModel(torch.nn.Module):
    def __init__(self, model_id, hf_auth, max_length=512):
        super(LlamaModel, self).__init__()
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_auth)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = self.load_model(model_id, hf_auth)

    def load_model(self, model_id, hf_auth):
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type='nf4',
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_compute_dtype=torch.bfloat16
                                        )
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     quantization_config=bnb_config,
                                                     trust_remote_code=True,
                                                     token=hf_auth,
                                                     use_safetensors=True,
                                                     device_map="auto"
                                                     )
        return model

    def forward(self, source, targets=None):
        input_ids = self.tokenizer.encode(source, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(input_ids.to(torch.long), do_sample=True, max_new_tokens=512, num_beams=2)
        generated_text = self.tokenizer.decode(generated_ids[0, len(input_ids.input_ids[0]):], skip_special_tokens=True).strip()
        generated_text = re.sub('\n+', '\n', generated_text)  # remove excessive newline characters

        return generated_text

class Inferencer:
    def __init__(self, model, prompt, testdata, task, write_dir):
        self.model = model
        self.prompt = prompt
        self.testdata = testdata
        self.task = task
        self.write_dir = write_dir

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

    def evaluate(self):
        # Evaluate the model's performance
        self.model.eval()
        results = []
        for batch_idx, inputs in enumerate(self.testdata):
            source = inputs.get('Source', None)
            targets = inputs.get('Target', None)
            if source:
                # Predict
                output = self.model([f"{self.prompt} \"\"\"{source}\"\"\" \nOutput:"])
                result = output.strip().replace('\n', ' ')
                results.append(result)
        # Define result directory path
        path = os.path.join(self.write_dir, f'{self.task}.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to the model")
    parser.add_argument("--model_path", help="path to the model") 
    parser.add_argument("--task", help="Training task")
    parser.add_argument("--data_path", help="path to the data")
    parser.add_argument("--write_path", help="path to write best model")
    args = parser.parse_args()

    task = args.task
    data_path = args.data_path
    model = args.model
    model_path = args.model_path
    # if "llama" in model:
    #     model_path = "llama"
    # elif "falcon" in model:
    #     model_path = "falcon"
    # else:
    #     print("Invalid model path")
    #     exit()

    write_path = os.path.join(f'{args.write_path}', f'{args.task}', f'{model_path}')
    # Create result directory if it doesn't exist.
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    dataset_dict = preprocess_data(data_path, task, model_path)
    train_dataset = CustomDataset(dataset_dict["train"])

    evaluation = {
        f"{task}_dev": dataset_dict["validation"],
        f"{task}_test": dataset_dict["test"],
        f"{task}_pipeline_eval": dataset_dict["pipeline_eval"],
        f"{task}_pipeline_test": dataset_dict["pipeline_test"]
    }

    prompt_dir = os.path.join(f'{args.write_path}', f'{args.task}', f'{task}_prompts.txt')
    prompt = read_file(prompt_dir)
    # print(prompt)

    model_id = model
    hf_auth = hf_token
    max_length = 1024
    llama_model = LlamaModel(model_id, hf_auth, max_length)

    # Feed the chatgpt the dev, test, and pipeline datasets for inference
    for dataset_name, dataset in evaluation.items():
        print(f"Evaluating {model_path} {dataset_name}.txt dataset:")
        inf = Inferencer(llama_model, prompt, dataset, task, write_path)
        inf.evaluate()

    t_task = {
    "ordering": "order",
    "structuring": "structure",
    "lexicalization": "lexicalized format",
    "reg": "refering expression generation format",
    "sr": "textual realization format"
    }

query1 = '''[TRIPLE] Italy capital Rome [/TRIPLE] [TRIPLE] A.S._Gubbio_1910 ground Italy [/TRIPLE] [TRIPLE] Italy language Italian_language [/TRIPLE] [TRIPLE] Italy leader Sergio_Mattarella [/TRIPLE]'''
query2 = '''<TRIPLE> Alan_Shepard awards Distinguished_Service_Medal_(United_States_Navy) </TRIPLE> <TRIPLE> Alan_Shepard birthPlace New_Hampshire </TRIPLE> <TRIPLE> Alan_Shepard deathDate "1998-07-21" </TRIPLE> <TRIPLE> Alan_Shepard deathPlace California </TRIPLE> <TRIPLE> Distinguished_Service_Medal_(United_States_Navy) higher Department_of_Commerce_Gold_Medal </TRIPLE> <TRIPLE> Alan_Shepard was_awarded "American_Defense_Service_ribbon.svg" </TRIPLE>
'''.replace('<', '[').replace('>', ']')
query3 = '''<TRIPLE> Turkey capital Ankara </TRIPLE> <TRIPLE> Atatürk_Monument_(İzmir) designer Pietro_Canonica </TRIPLE> <TRIPLE> Atatürk_Monument_(İzmir) inaugurationDate "1932-07-27" </TRIPLE> <TRIPLE> Turkey leaderName Ahmet_Davutoğlu </TRIPLE> <TRIPLE> Atatürk_Monument_(İzmir) location Turkey </TRIPLE> <TRIPLE> Atatürk_Monument_(İzmir) material "Bronze" </TRIPLE>
'''.replace('<', '[').replace('>', ']')
train_examples = '''
Example 1: """
[TRIPLE] Beef_kway_teow country "Singapore_and_Indonesia" [/TRIPLE] [TRIPLE] Beef_kway_teow ingredient Palm_sugar [/TRIPLE] [TRIPLE] Beef_kway_teow region Singapore [/TRIPLE]
"""
Output: ingredient country
###
Example 2: """
[TRIPLE] Bacon_Explosion country United_States [/TRIPLE] [TRIPLE] Bacon_Explosion ingredient Bacon [/TRIPLE] [TRIPLE] Bacon_Explosion mainIngredients "Bacon,sausage" [/TRIPLE] [TRIPLE] Bacon_Explosion region Kansas_City_metropolitan_area [/TRIPLE]
"""
Output: mainIngredients region country
###
Example 3: """
[TRIPLE] Adare_Manor architect "James_Pain_and_George_Richard_Pain," [/TRIPLE] [TRIPLE] Adare_Manor completionDate 1862 [/TRIPLE] [TRIPLE] Adare_Manor owner J._P._McManus [/TRIPLE]
"""
Output: architect completionDate owner
###
Example 4: """
[TRIPLE] Ajoblanco country Spain [/TRIPLE] [TRIPLE] Ajoblanco ingredient Garlic [/TRIPLE] [TRIPLE] Ajoblanco mainIngredients "Bread,_almonds,_garlic,_water,_olive_oil" [/TRIPLE] [TRIPLE] Ajoblanco region Andalusia [/TRIPLE]
"""
Output: region country mainIngredients
###
Example 5:"""
[TRIPLE] Asilomar_Conference_Grounds added_to_the_National_Register_of_Historic_Places "1987-02-27" [/TRIPLE] [TRIPLE] Asilomar_Conference_Grounds architecture "Arts_and_Crafts_Movement_and_American_craftsman_Bungalows" [/TRIPLE] [TRIPLE] Asilomar_Conference_Grounds location "Asilomar_Blvd.,_Pacific_Grove,_California" [/TRIPLE] [TRIPLE] Asilomar_Conference_Grounds yearOfConstruction 1913 [/TRIPLE]
"""
Output: location yearOfConstruction added_to_the_National_Register_of_Historic_Places
###'''

examples = f"""I would like to arrange my triples in a specific {t_task[task]} to control the way information is expressed in the final summary. 
Below, you'll find examples from my {task} dataset along with inputs and expected outputs:{train_examples}
Now strictly generate only the output result for this query, extra comments is not allowed: \n
Query: """

    input = f"{examples} \"\"\"{query2}\"\"\" \nOutput:"
    input2 = f"Ordering: {train_examples} \"\"\"{query1}\"\"\" \nOutput:"

    generated_text = llama_model(input)
    print("Generated Text:", generated_text)