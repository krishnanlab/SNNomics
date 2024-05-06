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
    def __init__(self, training_json, expression_mat):
        # used to prepare the labels and images path
        self.training_json = training_json
        self.expression_mat = expression_mat

    def __getitem__(self, index):
        anchor_ind = self.train_dir / self.train_df.iat[index, 0]
        pos_ind = self.train_dir / self.train_df.iat[index, 1]
        neg_ind = self.train_dir / self.train_df.iat[index, 2]

        anchor = self.expression_mat[anchor_ind, :]
        pos = self.expression_mat[pos_ind, :]
        neg = self.expression_mat[neg_ind, :]

        return anchor, pos, neg

    def __len__(self):
        return len(self.train_df)


class CVSplit:
    def __init__(self, labels: pd.DataFrame, k: int):
        self.labels = labels
        self.k = k
        self.folds = {}
        self.holdout = None

    def holdout_split(self, holdout_percent: float, seed: int):
        random.seed(seed)
        num_samples = len(self.labels)
        n_holdout_samples = int(num_samples * holdout_percent // 1)
        indices = random.sample(range(num_samples), n_holdout_samples)
        self.holdout = self.labels.index.to_numpy()[indices]
        self.labels = self.labels.drop(self.holdout)    # Remove holdout from labels

    # def generate_triplets(self):
    #     triplets = []
    #     for term in tqdm(self.labels.columns):
    #         positives = self.labels[self.labels[term] == 1].index.tolist()
    #         negatives = self.labels[self.labels[term] == -1].index
    #
    #         # Check for all positive or all negative
    #         if len(positives) < 2:
    #             print(f'Not enough positives to generate triplets for {term}.')
    #             continue
    #
    #         # Generate all possible combinations of two positive samples
    #         for anchor, pos in combinations(positives, 2):
    #             # Generate triplets
    #             for neg in negatives:
    #                 triplets.append((anchor, pos, neg))
    #
    #     return triplets

    @staticmethod
    def generate_triplets_for_term(args):
        term, labels = args
        triplets = []
        positives = labels[labels[term] == 1].index.tolist()
        negatives = labels[labels[term] == -1].index

        if len(positives) < 2:
            print(f'Not enough positives to generate triplets for {term}.')
            return []

        for anchor, pos in combinations(positives, 2):
            for neg in negatives:
                triplets.append((anchor, pos, neg))

        return triplets

    def generate_triplets(self):
        triplets = []
        num_cores = multiprocessing.cpu_count()

        with multiprocessing.Pool(num_cores) as pool:
            results = list(tqdm(pool.imap(
                self.generate_triplets_for_term,
                [(term, self.labels) for term in self.labels.columns]
            ), total=len(self.labels.columns), desc="Generating triplets"))

        for result in results:
            triplets.extend(result)

        return triplets

    def generate_dataset(self, df):
        triplets = self.generate_triplets()
        dataset = []
        for triplet in triplets:
            anchor, positive, negative = triplet
            dataset.append((anchor, positive, negative, 1))
            dissimilar_negative = np.random.choice(df.index[df.index != anchor])
            dataset.append((anchor, negative, dissimilar_negative, 0))
        return pd.DataFrame(dataset, columns=['Anchor', 'Positive', 'Negative', 'Label'])

    def k_fold_cv(self, seed):
        kf = KFold(n_splits=self.k, shuffle=True, random_state=seed)
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
        folds_file = outdir / 'folds.json'

        np.savetxt(holdout_file, self.holdout)  # Save holdout
        with open(folds_file, 'w') as f:
            json.dump(self.folds, f, indent=4)  # Save folds

    def check_uniformity(self):
        a = self.labels.to_numpy()
        is_unique = (a[0] == a).all(0)

        return