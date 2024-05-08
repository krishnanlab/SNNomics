import gzip
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from itertools import combinations
from torch.utils.data import Dataset
from sklearn.model_selection import KFold


class SiameseDataset(Dataset):
    """ Loads samples from an expression matrix to be used for training

    Attributes
    ----------
    training_json: json
        A json file containing

    expression_mat: np.array
        A samples x genes matrix of expression values
    """
    def __init__(self,
                 expression_mat: np.ndarray,
                 gsms: np.array,
                 training_json: json,
                 fold: int,
                 split: str
        ):
        self.training_json = training_json
        self.expression_mat = expression_mat
        self.gsms = gsms
        self.fold = fold
        self.split = split

    def __getitem__(self, index):
        samples = self.training_json[self.fold][self.split]
        anchor_gsm = samples[index][0]
        pos_gsm = samples[index][1]
        neg_gsm = samples[index][2]

        anchor_ind = np.where(np.isin(self.gsms, anchor_gsm))[0]
        pos_ind = np.where(np.isin(self.gsms, pos_gsm))[0]
        neg_ind = np.where(np.isin(self.gsms, neg_gsm))[0]
        
        print(anchor_ind)
        print(pos_ind)
        print(neg_ind)

        anchor = self.expression_mat[anchor_ind, :]
        pos = self.expression_mat[pos_ind, :]
        neg = self.expression_mat[neg_ind, :]

        return anchor, pos, neg

    def __len__(self):
        return len(self.training_json[self.fold][self.split])


class CVSplit:
    def __init__(self, labels: pd.DataFrame, k: int, triplet_margin: int, seed: int):
        self.labels = labels
        self.k = k
        self.triplet_margin = triplet_margin
        self.seed=seed
        self.folds = {}
        self.holdout = None

    def holdout_split(self, holdout_percent: float):
        random.seed(self.seed)
        num_samples = len(self.labels)
        n_holdout_samples = int(num_samples * holdout_percent // 1)
        indices = random.sample(range(num_samples), n_holdout_samples)
        self.holdout = self.labels.index.to_numpy()[indices].astype(str)
        self.labels = self.labels.drop(self.holdout)    # Remove holdout from labels

    @staticmethod
    def generate_triplets_for_term(args):
        term, labels, triplet_margin, seed = args
        random.seed(seed)
        triplets = []
        positives = labels[labels[term] == 1].index.tolist()
        negatives = labels[labels[term] == -1].index

        if len(positives) < 2:
            print(f'Not enough positives to generate triplets for {term}.')
            return []

        combinations_list = list(combinations(positives, 2))
        
        if len(combinations_list) < triplet_margin:
            combinations_subset = random.sample(combinations_list, len(combinations_list))
        else:
            combinations_subset = random.sample(combinations_list, triplet_margin)
        for anchor, pos in combinations_subset:
            for i, neg in enumerate(negatives):
                triplets.append((anchor, pos, neg))
                if i == 2:
                    continue

        return triplets

    def generate_triplets(self, df):
        triplets = []
        num_cores = multiprocessing.cpu_count()

        with multiprocessing.Pool(num_cores) as pool:
            results = list(tqdm(pool.imap(
                self.generate_triplets_for_term,
                [(term, df, self.triplet_margin, self.seed) for term in df.columns]
            ), total=len(df.columns), desc="Generating triplets"))

        for result in results:
            triplets.extend(result)

        return triplets

    def generate_dataset(self, df):
        triplets = self.generate_triplets(df)
        dataset = []
        for triplet in triplets:
            anchor, positive, negative = triplet
            dataset.append((anchor, positive, negative))
        return pd.DataFrame(dataset, columns=['Anchor', 'Positive', 'Negative'])

    def k_fold_cv(self):
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        for i, (train_index, test_index) in enumerate(kf.split(self.labels)):
            print(f"Generating fold {i+1}")
            train_df, test_df = self.labels.iloc[train_index], self.labels.iloc[test_index]
            train_dataset = self.generate_dataset(train_df)
            test_dataset = self.generate_dataset(test_df)
            self.folds[f"{i + 1}"] = {
                "train": train_dataset.values.tolist(),
                "test": test_dataset.values.tolist()
            }

    def save(self, outdir: Path):
        print(f'Saving datasets to {outdir}')
        holdout_file = outdir / 'holdout.txt'
        folds_file = outdir / 'folds.json.gz'

        np.savetxt(holdout_file, self.holdout, fmt='%s')  # Save holdout
        json_str = json.dumps(self.folds, indent=4)  # Save folds
        json_bytes = json_str.encode('utf-8')
        with gzip.open(folds_file, 'w') as f:
            f.write(json_bytes)
