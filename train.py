import gzip
import json
import time
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from SNNomics.dataset import SiameseDataset
from SNNomics.model import SNN
from SNNomics.utils import check_dir
from SNNomics.trainer import Trainer

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-epochs',
        help='number of epochs to run',
        type=int,
        default=25,
    )
    parser.add_argument(
        '-folds',
        help='Path to json file containing training folds',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-expression_mat',
        help='Path to .npz file containing a samples x genes expression matrix',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--save',
        help='if applied, will save the model weights',
        action='store_true',
    )
    parser.add_argument(
        '-batch_size',
        help='size of each batch',
        type=int,
        default=128,
    )
    parser.add_argument(
        '-lr',
        help='learning rate',
        type=float,
        default=0.001,
    )
    parser.add_argument(
        '-weight_decay',
        help='weight_decay',
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        '-outdir',
        help='directory to save results to',
        type=str,
        default='results',
    )
    parser.add_argument(
        '-out_prefix',
        help='prefix of results outfiles',
        type=str,
        default=None,
    )
    args = parser.parse_args()
    start = time.time()

    # Set paths
    folds_file = Path(args.folds)
    expression_file = Path(args.expression_mat)
    outdir = Path(args.outdir)
    check_dir(outdir)

    # Assign training variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    epochs = args.epochs
    save_model = args.save
    lr = args.lr
    weight_decay = args.weight_decay

    # Load data
    expression_data = np.load(expression_file, allow_pickle=True)
    expression = expression_data['expression']
    samples = expression_data['gsms']
    genes = expression_data['genes']
    num_genes = len(genes)

    # Load folds
    with gzip.open(folds_file, 'r') as f:        
        folds_bytes = f.read()                     

    folds_str = folds_bytes.decode('utf-8')
    folds = json.loads(folds_str) 

    # Train model with k-fold CV
    for k in folds:
        print(f"Training fold {k}")
        # Assign training arguments
        model = SNN(num_genes)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        train_data = SiameseDataset(expression_mat=expression, gsms=samples, training_json=folds, fold=k, split='train')
        test_data = SiameseDataset(expression_mat=expression, gsms=samples, training_json=folds, fold=k, split='test')
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=6, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=6, shuffle=True)

        trainer = Trainer(
            model,
            optimizer,
            triplet_loss,
            train_loader,
            test_loader,
            device,
            outdir,
            k,
        )
        trainer.train(epochs=epochs)
        trainer.test()
        trainer.save_weights(f'model_fold-{k}.pt')
