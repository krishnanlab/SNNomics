import numpy as np
import pandas as pd
import torch
import torch.nn
from pathlib import Path
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, test_loader, device, outdir, k):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.outdir = outdir
        self.k = k
        self.train_loss = {'epoch': [], 'loss': []}
        self.test_loss = []

    def train(self, epochs: int):
        for i in range(epochs):
            print('Epoch {}/{}'.format(i + 1, epochs))
            run_result = {'nsamples': 0, 'loss': 0}
            for data in tqdm(self.train_loader):
                a, p, n = data
                self.optimizer.zero_grad()
                
                for param in self.model.parameters():
                    if param.grad is not None:
                        del param.grad  # free some memory
                torch.cuda.empty_cache()

                anchor, positive, negative = a.to(self.device), p.to(self.device), n.to(self.device)

                output_anchor, output_positive, output_negative = self.model(anchor.float(), positive.float(), negative.float())

                loss = self.criterion(output_anchor, output_positive, output_negative)

                loss.backward()
                self.optimizer.step()

                run_result['nsamples'] += a.size(0)
                run_result['loss'] += loss.item()
                
                if i == epochs-1:
                    self.save_loss('train')

            self.train_loss['epoch'].append(i+1)
            self.train_loss['loss'].append(run_result['loss'] / run_result['nsamples'])
    
    def test(self):
        with torch.no_grad():
            for a, p, n in self.test_loader:
                anchor, positive, negative = a.to(self.device), p.to(self.device), n.to(self.device)
                output_anchor, output_positive, output_negative = self.model(anchor.float(), positive.float(), negative.float())
                loss = self.criterion(output_anchor, output_positive, output_negative)
                self.test_loss.append(loss.item())
        print(f'Average test loss at fold {self.k}: {np.mean(self.test_loss)}')

    def save_weights(self, outfile: str):
        torch.save(self.model.state_dict(), self.outdir / outfile)

    def save_loss(self, _type):
        if _type == 'train':
            df = pd.DataFrame.from_dict(self.train_loss)
            df.to_csv(self.outdir / f'{_type}_loss_fold-{self.k}.csv', index=False)
        elif _type == 'test':
            np.save(self.outdir / f'{_type}_loss_fold-{self.k}.npy')

