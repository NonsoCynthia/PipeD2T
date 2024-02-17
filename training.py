import os
# Set the environment variable to disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import nltk
import torch
from torch import optim
import pytorch_lightning as pl
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
nltk.download('punkt')
import wandb
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
pl.seed_everything(42)

# Create a Trainer class based on PyTorch Lightning's LightningModule
class Trainer(pl.LightningModule):
    def __init__(self, model, trainloader, devdata, optimizer, epochs, batch_status, write_path, early_stop=5, verbose=True):
        # Initialize the Trainer class
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_status = batch_status
        self.early_stop = early_stop
        self.verbose = verbose
        self.trainloader = trainloader
        self.devdata = devdata
        self.write_path = write_path
        
        if not os.path.exists(write_path):
            os.mkdir(write_path)

        # Initialize WandB
        # wandb_project = 'webnlg'  # args.wandb_project
        # wandb_entity = 'afrisent-nlp'  # args.wandb_entity
        # wandb.init(project=wandb_project, entity=wandb_entity)

    def train(self):
        # Train the model
        max_bleu, repeat = 0, 0
        for epoch in range(self.epochs):
            # For the given number of epochs train the number
            self.model.model.train() #sets the model in evaluation mode using train function
            losses = []
            for batch_idx, inputs in enumerate(self.trainloader):
                source, targets = inputs['Source'], inputs['Target']
                self.optimizer.zero_grad()

                # generating
                output = self.model(source, targets)

                # Calculate loss
                loss = output.loss
                losses.append(float(loss))

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                # Log loss to WandB
                # wandb.log({'train_loss': loss})

                # Display training progress
                if (batch_idx + 1) % self.batch_status == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal Loss: {:.6f}'.format(epoch, \
                            batch_idx + 1,len(self.trainloader),100. * batch_idx / len(self.trainloader), \
                            float(loss),round(sum(losses) / len(losses),5)))

            bleu = self.evaluate()
            print("Model: ", '-'.join(str(self.write_path).split('/')[-2:]), 'BLEU: ', bleu)
            # wandb.log({'bleu_score': bleu})

            if bleu > max_bleu:
                self.model.model.save_pretrained(os.path.join(self.write_path, 'model'))
                # wandb.save(self.write_path)
                max_bleu = bleu
                repeat = 0
                print('Saving best model...')
            else:
                repeat += 1

            if repeat == self.early_stop:
                break

    def evaluate(self):
        # Evaluate the model's performance
        self.model.model.eval() #sets the model in evaluation mode using eval function
        results = {}
        for batch_idx, inputs in enumerate(self.devdata):
            source, targets = inputs['Source'], inputs['Target']
            if source not in results:
                # Initialize the dictionary for this source
                # hyp is the generated texts; refs is the original targets
                results[source] = {'hyp': '', 'refs': []}

                # Predict
                output = self.model([source])
                results[source]['hyp'] = output[0]

                # Display evaluation progress
                if (batch_idx + 1) % self.batch_status == 0:
                    print('Evaluation: [{}/{} ({:.0f}%)]'.format(batch_idx + 1, \
                                len(self.devdata), 100. * batch_idx / len(self.devdata)))

            # Store references as a list of lists
            results[source]['refs'].append(targets)

        hypothesis, references = [], []
        
        for source in results.keys():
            # if self.verbose:
                # print('Source:', source)
                # for ref in results[source]['refs']:
                    # print('Real: ', ref)
                # print('Pred: ', results[source]['hyp'])
                # print()

            # Tokenize hypotheses and references
            hypothesis.append(nltk.word_tokenize(results[source]['hyp']))
            references.append([nltk.word_tokenize(ref) for ref in results[source]['refs']])

        chencherry = SmoothingFunction()
        bleu = corpus_bleu(references, hypothesis, smoothing_function=chencherry.method3)
        return bleu