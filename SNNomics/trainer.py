import torch
import torch.nn


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def train(self, epochs):
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
                loss.backward()
                self.optimizer.step()
