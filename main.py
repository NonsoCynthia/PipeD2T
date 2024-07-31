import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, Dataset
from data.load_dataset import CustomDataset, preprocess_data, read_file, read_dict
from models.BART import BART_Model
from models.GPT2 import GPT2_Model
from models.T5 import T5_Model
from models.LLAMA import LLAMA_Model
from models.FALCON import FALCON_Model
from training import Trainer
#from training_llama import Trainer_llama
#import pytorch_lightning as pl
from transformers import set_seed
set_seed(42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", help="path to the tokenizer")
    parser.add_argument("--model_path", help="path to the model")
    parser.add_argument("--task", help="Traing task")
    parser.add_argument("--data_path", help="path to the data")
    parser.add_argument("--epochs", help="number of epochs", type=int)
    parser.add_argument("--learning_rate", help="learning rate", type=float)
    parser.add_argument("--train_batch_size", help="batch size of training", type=int)
    parser.add_argument("--early_stop", help="earling stop", type=int)
    parser.add_argument("--max_length", help="maximum length to be processed by the network", type=int)
    parser.add_argument("--write_path", help="path to write best model")
    parser.add_argument("--verbose", help="should display the loss?", action="store_true")
    parser.add_argument("--batch_status", help="display of loss", type=int)
    # parser.add_argument("--cuda", help="use CUDA", action="store_true")

    args = parser.parse_args()    

    # Model settings, Settings and configurations
    tokenizer_path = args.tokenizer
    model_path = args.model_path
    task = args.task
    data = args.data_path
    epochs = args.epochs
    learning_rate = args.learning_rate
    train_batch_size = args.train_batch_size
    early_stop = args.early_stop
    max_length = args.max_length
    write_path = args.write_path
    verbose = args.verbose if 'verbose' in args else False
    batch_status = args.batch_status
    # device = torch.device('cuda' if args.cuda else 'cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # input_path = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/Data/deepnlg/"
    # task = "ordering"  # Replace with your task handling logic

    # Create model
    if 'bart' in tokenizer_path:
        mod = 'bart'
        write_path = os.path.join(write_path, f"{task}/{mod}") 
        generator = BART_Model(tokenizer_path, model_path, max_length, sep_token=task+":")
    elif 't5' in tokenizer_path:
        mod = 'flan-t5-large' if 'flan' in tokenizer_path else 't5-large' #if 'base' in tokenizer_path else 'flan-t5-large'
        write_path = os.path.join(write_path, f"{task}/{mod}")
        generator = T5_Model(tokenizer_path, model_path, max_length, sep_token=task+":")
    elif 'gpt2' in tokenizer_path:
        mod = 'gpt2'
        write_path = os.path.join(write_path, f"{task}/{mod}")
        generator = GPT2_Model(tokenizer_path, model_path, max_length, sep_token=task+":")
    elif 'llama' in tokenizer_path:
        mod = 'llama'
        write_path = os.path.join(write_path, f"{task}/{mod}")
        generator = LLAMA_Model(tokenizer_path, model_path, max_length, sep_token=task+":")
    elif 'falcon' in tokenizer_path:
        mod = 'falcon'
        write_path = os.path.join(write_path, f"{task}/{mod}")
        generator = FALCON_Model(tokenizer_path, model_path, max_length, sep_token=task+":")
    else:
        raise Exception("Invalid model")
   
    dataset_dict = preprocess_data(data, task, mod)
    train_dataset = CustomDataset(dataset_dict["train"])
    validation_dataset = CustomDataset(dataset_dict["validation"])
    test_dataset = dataset_dict["test"]

    # Create data loader
    trainloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True) #num_workers=10
    #validation_dataset = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    # Create optimizer
    optimizer = torch.optim.AdamW(generator.model.parameters(), lr=learning_rate)
    
    # Trainer settings
    trainer = Trainer(generator, trainloader, validation_dataset, optimizer, epochs, batch_status, write_path, early_stop, verbose)

    # Train the model
    trainer.train()
