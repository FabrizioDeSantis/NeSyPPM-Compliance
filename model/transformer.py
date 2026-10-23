import torch
import torch.nn as nn

from model.inputs import EventEncoder


class EventTransformer(nn.Module):
    '''
    Transformer encoder over the events of a prefix, mean-pooled over the non-padding events.
    Returns P(label = 1), shape (B, 1).
    With num_rule_features > 0 the input also contains the rule truth values (see EventEncoder).
    '''

    def __init__(self, feature_names, vocab_sizes, seq_len, model_dim=64, num_layers=2, num_heads=4, dropout=0.1,
                 num_rule_features=0):
        super().__init__()
        self.encoder = EventEncoder(feature_names, vocab_sizes, seq_len, model_dim, num_rule_features)
        self.input_proj = nn.Linear(self.encoder.output_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1)
        )

    def forward(self, x):
        events, padding = self.encoder(x)
        h = self.input_proj(events) + self.positional_encoding
        h = self.transformer(h, src_key_padding_mask=padding)
        keep = (~padding).unsqueeze(-1).float()
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        return torch.sigmoid(self.classifier(pooled))
