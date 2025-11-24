import torch
import torch.nn as nn
import torch.nn.functional as F

class EventTransformerA(nn.Module):
    def __init__(self, vocab_sizes, config, feature_names, numerical_features, model_dim, num_classes, max_len, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.config = config
        self.feature_names = feature_names
        self.numerical_features = numerical_features
        torch.manual_seed(42)
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
            for feature, vocab_size in vocab_sizes.items()
        })
        transformer_input_size = (model_dim * len(self.embeddings)) + len(self.numerical_features) + 3
        self.input_proj = nn.Linear(transformer_input_size, model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        classifier_input_size = model_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, classifier_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size // 2, num_classes)
        )

        self.sigmoid = nn.Sigmoid()
    
    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = self.config.sequence_length
        embeddings_list = []
        numerical_features = []
        # Process categorical features
        for name in self.embeddings.keys():
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx].long()
            embeddings_list.append(self.embeddings[name](feature_data))
            
        # Process numerical features
        for name in self.numerical_features:
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx]
            numerical_features.append(feature_data)
        # sepsis
        if self.config.dataset == 'sepsis':
            numerical_features.append(x[:, 364:377])
            numerical_features.append(x[:, 377:390])
            numerical_features.append(x[:, 390:403])
        # bpi12
        elif self.config.dataset == 'bpi12':
            numerical_features.append(x[:, 120:160])
            numerical_features.append(x[:, 160:200])
            numerical_features.append(x[:, 200:240])
        # traffic
        elif self.config.dataset == 'traffic':
            numerical_features.append(x[:, 110:120])
            numerical_features.append(x[:, 120:130])
            numerical_features.append(x[:, 130:140])
        # bpi17
        elif self.config.dataset == 'bpi17':
            numerical_features.append(x[:, 240:260])
            numerical_features.append(x[:, 260:280])
            numerical_features.append(x[:, 280:300])
        numerical_features = torch.stack(numerical_features, dim=2)
        output = torch.cat(embeddings_list + [numerical_features], dim=2)
        return output
    
    def forward(self, x, mask=None):

        x = self._get_embeddings(x)
        x = self.input_proj(x)
        x = x + self.positional_encoding[:, :x.size(1), :]

        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        pooled = x.mean(dim=1)

        output = self.classifier(pooled)


        return self.sigmoid(output)
    
class EventTransformer(nn.Module):
    def __init__(self, vocab_sizes, config, feature_names, numerical_features, model_dim, num_classes, max_len, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.config = config
        self.feature_names = feature_names
        self.numerical_features = numerical_features
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
            for feature, vocab_size in vocab_sizes.items()
        })
        transformer_input_size = (model_dim * len(self.embeddings)) + len(self.numerical_features)
        self.input_proj = nn.Linear(transformer_input_size, model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        classifier_input_size = model_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, classifier_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size // 2, num_classes)
        )

        self.sigmoid = nn.Sigmoid()
    
    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = self.config.sequence_length
        embeddings_list = []
        numerical_features = []
        # Process categorical features
        for name in self.embeddings.keys():
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx].long()
            embeddings_list.append(self.embeddings[name](feature_data))
            
        # Process numerical features
        for name in self.numerical_features:
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx]
            numerical_features.append(feature_data)
        numerical_features = torch.stack(numerical_features, dim=2)
        output = torch.cat(embeddings_list + [numerical_features], dim=2)
        return output
    
    def forward(self, x, mask=None):

        x = self._get_embeddings(x)
        x = self.input_proj(x)
        x = x + self.positional_encoding[:, :x.size(1), :]

        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        pooled = x.mean(dim=1)

        output = self.classifier(pooled)


        return self.sigmoid(output)