import json
import time
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from SNNomics.model import SNN
from SNNomics.utils import check_dir
from SNNomics.dataset import PredictDataset
from SNNomics.predictor import Predictor


def rm_query(query_id: str, database: np.ndarray, database_ids: np.array):
    query_ind = np.where(database_ids == query_id)[0]
    query_vector = database[query_ind, :]
    database_queryRm = np.delete(databse, query_ind)
    ids_queryRm = np.delete(database_ids, query_ind)
    return query_vector, database_queryRm, ids_queryRm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-query',
        help='Path to .txt file containing GSMs to predict',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-database',
        help='Path to .npz file containing a samples x genes expression matrix',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-batch_size',
        help='size of each batch',
        type=int,
        default=128,
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

    # Set paths
    samples_file = Path(args.samples)
    database_file = Path(args.database)
    outdir = Path(args.outdir)
    check_dir(outdir)

    # Load data
    data = np.load(database_file)
    database = data['expression']
    database_ids = data['gsms']
    genes = data['genes']

    # Remove query from database
    query, database, database_ids = rm_query(args.query, database)
    
    # Predict for queries
    predict_data = PredictDataset(database, database_ids)
    loader = DataLoader(predict_data, batch_size=batch_size, num_workers=6, shuffle=False)
    predictor = Predictor(
        query,
        model,
        criterion,
        loader,
        device,
        outdir,
    )
    
