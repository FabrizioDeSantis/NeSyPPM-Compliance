import torch
import torch.nn as nn


class EventEncoder(nn.Module):
    '''
    Turns the flat model input into one vector per event.

    The input x has shape (B, F * L + R): the F features of a prefix, each padded to L events and stored
    one after the other (feature-major, see data/preprocess_*.create_ngrams), followed by R rule truth
    values (input of the feat_eng / A configurations, see nesy/knowledge.py; R = 0 otherwise).

    forward(x) returns
        events:  (B, L, D) an embedding per categorical feature (those in vocab_sizes), the numerical
                 features as they are and the R rule values, repeated at every event
        padding: (B, L) True on padding positions (activity code 0)
    '''

    def __init__(self, feature_names, vocab_sizes, seq_len, embedding_dim, num_rule_features=0,
                 activity_feature="concept:name"):
        super().__init__()
        self.seq_len = seq_len
        self.num_rule_features = num_rule_features
        offset = {name: i * seq_len for i, name in enumerate(feature_names)}
        self.categorical = [(name, offset[name]) for name in feature_names if name in vocab_sizes]
        self.numerical = [offset[name] for name in feature_names if name not in vocab_sizes]
        self.activity_offset = offset[activity_feature]
        self.rules_offset = len(feature_names) * seq_len
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(int(vocab_sizes[name]) + 1, embedding_dim, padding_idx=0)
            for name, _ in self.categorical
        })
        self.output_dim = embedding_dim * len(self.categorical) + len(self.numerical) + num_rule_features

    def _feature(self, x, offset):
        return x[:, offset:offset + self.seq_len]

    def forward(self, x):
        parts = [self.embeddings[name](self._feature(x, offset).long()) for name, offset in self.categorical]
        if self.numerical:
            parts.append(torch.stack([self._feature(x, offset) for offset in self.numerical], dim=2))
        if self.num_rule_features:
            rules = x[:, self.rules_offset:self.rules_offset + self.num_rule_features]
            parts.append(rules.unsqueeze(1).expand(-1, self.seq_len, -1))
        padding = self._feature(x, self.activity_offset) == 0
        return torch.cat(parts, dim=2), padding
