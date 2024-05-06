import argparse
import pandas as pd
from pathlib import Path
from SNNomics.dataset import CVSplit
from SNNomics.utils import check_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-labels',
        help='/path/to/labels.csv with GSM/GSE as index and term names as the columns',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-holdout_percent',
        help='percentage of dataset to be used for final test set',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '-outdir',
        help='/path/to/outdir to save the folds dataframes',
        type=str,
        default='data',
    )
    args = parser.parse_args()

    # Set paths
    labels_file = Path(args.labels)
    outdir = Path(args.outdir)
    check_dir(outdir)   # Make outdir if it doesn't exist

    # Load labels
    labels = pd.read_csv(labels_file, index_col=0)

    # Generate holdout split and k-fold CV splits
    splitter = CVSplit(labels=labels, k=3)
    splitter.holdout_split(holdout_percent=args.holdout_percent, seed=22)    # Generate final holdout before k-fold CV
    splitter.k_fold_cv(seed=22)
    splitter.save(outdir)
