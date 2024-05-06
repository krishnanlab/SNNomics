import numpy as np
import pandas as pd
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
    def __init__(self, training_json=None, expression_mat=None):
        # used to prepare the labels and images path
        self.train_df = training_json
        self.train_df.columns = ["image1", "image2", "label"]
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
        self.folds = []

    def generate_triplets(self):
        triplets = []
        for group in self.labels.columns:
            # Selecting positive samples
            positives = self.labels[self.labels[group] == 1].index.tolist()
            # Selecting negative samples
            negatives = self.labels[self.labels[group] == -1].index.tolist()
            # Generating triplets
            for anchor in positives:
                for pos in positives:
                    if anchor != pos:
                        for neg in negatives:
                            triplets.append((anchor, pos, neg))
        return triplets

    def generate_dataset(self, df):
        triplets = self.generate_triplets()
        dataset = []
        for triplet in triplets:
            anchor, positive, negative = triplet
            dataset.append((anchor, positive, negative, 1))  # Anchor and positive are similar
            # Randomly select a negative sample to make it dissimilar
            dissimilar_negative = np.random.choice(df.index[df.index != anchor])
            dataset.append((anchor, negative, dissimilar_negative, 0))  # Anchor and negative are dissimilar
        return pd.DataFrame(dataset, columns=['Anchor', 'Positive', 'Negative', 'Label'])

    def k_fold_cv(self, seed=22):
        kf = KFold(n_splits=self.k, shuffle=True, random_state=seed)
        for train_index, test_index in kf.split(self.labels):
            train_df, test_df = self.labels.iloc[train_index], self.labels.iloc[test_index]
            train_dataset = self.generate_dataset(train_df)
            test_dataset = self.generate_dataset(test_df)
            self.folds.append((train_dataset, test_dataset))

    def save(self, outdir: Path):
        for i, (train_data, test_data) in enumerate(self.folds):
            train_data.to_csv(outdir / f'train_fold_{i + 1}.csv', index=False)
            test_data.to_csv(outdir / f'test_fold_{i + 1}.csv', index=False)
