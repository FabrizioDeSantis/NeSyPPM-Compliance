'''Datasets of the experiments: where they are, how they are preprocessed and how they are fed to the models.'''
import contextlib
from collections import Counter
from dataclasses import dataclass
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import preprocess_bpi12, preprocess_bpi17, preprocess_sepsis, preprocess_traffic
from data.dataset import NeSyDataset


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    preprocessing: ModuleType  # data/preprocess_<dataset>.py: split, encoding and prefixes
    epochs_ltn: int            # default number of LTN epochs (fewer on the large BPI logs)


DATASETS = {
    "bpi12": DatasetSpec("data_processed/bpi12.csv", preprocess_bpi12, epochs_ltn=5),
    "bpi17": DatasetSpec("data_processed/bpi17.csv", preprocess_bpi17, epochs_ltn=5),
    "sepsis": DatasetSpec("data_processed/sepsis_3.csv", preprocess_sepsis, epochs_ltn=15),
    "traffic": DatasetSpec("data_processed/traffic_fines.csv", preprocess_traffic, epochs_ltn=15),
}


@dataclass(frozen=True)
class FeatureLayout:
    '''Position of the features in the flat model input: feature i occupies columns [i * L, (i + 1) * L).'''
    feature_names: tuple
    seq_len: int

    @property
    def width(self):
        return len(self.feature_names) * self.seq_len

    def get(self, x, name):
        '''(B, L) values of feature `name` at every event of the prefixes in x (0 on padding).'''
        start = self.feature_names.index(name) * self.seq_len
        return x[:, start:start + self.seq_len]


@dataclass
class PreparedData:
    name: str
    raw: pd.DataFrame  # event log as read from disk, before preprocessing
    train: NeSyDataset
    val: NeSyDataset
    test: NeSyDataset
    layout: FeatureLayout
    vocab_sizes: dict  # categorical feature -> number of codes (codes start at 1, 0 is padding)
    scalers: dict      # numerical feature -> fitted MinMaxScaler
    events: pd.DataFrame = None  # encoded event log, as given to create_ngrams
    split_ids: dict = None       # "train" / "val" / "test" -> case ids


@dataclass
class TraceSet:
    '''Entire traces (all the events of each case), encoded like the prefixes, padded to the longest trace.'''
    x: torch.Tensor  # (N, F * L)
    y: torch.Tensor  # (N,)
    split: np.ndarray  # (N,) "train", "val" or "test"
    layout: FeatureLayout
    case_ids: np.ndarray  # (N,)


def load_dataset(name):
    '''Read and preprocess a dataset. The train / val / test split is fixed by the seeded preprocessing.'''
    spec = DATASETS[name]
    print(f"-- Reading {spec.path}")
    raw = pd.read_csv(spec.path, dtype={"org:resource": str})
    with capture_encoded_log(spec.preprocessing) as log:
        (X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, scalers = \
            spec.preprocessing.preprocess_eventlog(raw.copy())
    layout = FeatureLayout(tuple(feature_names), seq_len=len(X_train[0][0]))
    data = PreparedData(name, raw, NeSyDataset(X_train, y_train), NeSyDataset(X_val, y_val),
                        NeSyDataset(X_test, y_test), layout, vocab_sizes, scalers, log["events"], log["ids"])
    print_summary(data)
    return data


@contextlib.contextmanager
def capture_encoded_log(module):
    '''Record what preprocess_eventlog passes to module.create_ngrams: the encoded event log and the split case ids.'''
    captured = {}
    original = module.create_ngrams

    def create_ngrams(data, train_ids, val_ids, test_ids, *args, **kwargs):
        captured.update(events=data, ids={"train": list(train_ids), "val": list(val_ids), "test": list(test_ids)})
        return original(data, train_ids, val_ids, test_ids, *args, **kwargs)

    module.create_ngrams = create_ngrams
    try:
        yield captured
    finally:
        module.create_ngrams = original


def encode_traces(data):
    '''The entire trace of every case, in the feature-major layout of the prefixes (events in log order).'''
    events = data.events.reset_index(drop=True)
    case_index, case_ids = pd.factorize(events["case:concept:name"], sort=True)
    position = events.groupby(case_index).cumcount().to_numpy()
    names = list(data.layout.feature_names)
    x = np.zeros((len(case_ids), len(names), position.max() + 1), dtype=np.float32)
    x[case_index, :, position] = events[names].to_numpy(dtype=np.float32)
    y = events.groupby(case_index)["label"].first().to_numpy(dtype=np.float32)
    split_of = {case: split for split, ids in data.split_ids.items() for case in ids}
    split = np.array([split_of.get(case) for case in case_ids], dtype=object)
    layout = FeatureLayout(data.layout.feature_names, seq_len=x.shape[2])
    return TraceSet(torch.from_numpy(x).flatten(1), torch.from_numpy(y), split, layout, np.asarray(case_ids))


def print_summary(data):
    print(f"-- Dataset {data.name}: {len(data.layout.feature_names)} features x {data.layout.seq_len} events")
    print(f"   features: {list(data.layout.feature_names)}")
    for split in ("train", "val", "test"):
        dataset = getattr(data, split)
        counts = Counter(int(label) for label in dataset.labels.tolist())
        print(f"   {split}: {len(dataset)} prefixes, labels {dict(sorted(counts.items()))}")


def make_loader(dataset, batch_size, shuffle=False, seed=None):
    '''With shuffle, the batch order is fixed by seed.'''
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
