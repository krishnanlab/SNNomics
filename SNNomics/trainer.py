import pandas as pd
import torch
import torch.nn
from pathlib import Path


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, test_loader, device, outdir):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.outdir = outdir
        self.train_loss = {'epoch': [], 'loss': []}

    def train(self, epochs: int):
        for i in range(epochs):
            print('Epoch {}/{}'.format(i + 1, epochs))
            for a, p, n in self.train_loader:
                self.optimizer.zero_grad()

                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()

                anchor, positive, negative = a.to(self.device), p.to(self.device), n.to(self.device)

                output_anchor, output_positive = self.model(anchor, positive)
                output_negative = self.model(negative)

                loss = self.loss_fn(output_anchor, output_positive, output_negative)
                self.train_loss['loss'].append(loss.item())
                self.train_loss['epoch'].append(i+1)

                loss.backward()
                self.optimizer.step()

                if i == epochs-1:
                    self.save_loss()

    def test(self):
        with torch.no_grad():
            for a, p, n in self.test_loader:
                anchor, positive, negative = a.to(self.device), p.to(self.device), n.to(self.device)
                output_anchor, output_positive = self.model(anchor, positive)
                output_negative = self.model(negative)
                loss = self.loss_fn(output_anchor, output_positive, output_negative)

    def save_weights(self, outfile: str):
        torch.save(self.model.state_dict(), outfile)

    def save_loss(self):
        df = pd.DataFrame.from_dict(self.train_loss)
        df.to_csv(self.outdir / 'train_loss.csv', index=False)
