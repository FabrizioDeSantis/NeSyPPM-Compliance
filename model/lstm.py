import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration class for LSTM model parameters"""
    hidden_size: int
    dataset: str
    num_layers: int
    sequence_length: int
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    num_features: int = 4
    num_epochs: int = 100

class Attention(nn.Module):
    """Attention mechanism for Bi-LSTM"""
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        # Attention weights
        self.attention = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, lstm_output: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_size * 2)
            lengths: (batch_size,) actual lengths of sequences
        Returns:
            context: (batch_size, hidden_size * 2) weighted sum of lstm outputs
        """
        batch_size, seq_len, hidden_dim = lstm_output.size()
        
        attention_scores = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)
        
        mask = torch.arange(seq_len, device=lstm_output.device).unsqueeze(0) < lengths.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)  # (batch_size, 1, hidden_dim)
        context = context.squeeze(1)  # (batch_size, hidden_dim)
        
        return context
    
class LSTMModelWithAttention(nn.Module):
    def __init__(self, vocab_sizes: List[int], config, num_classes: int, feature_names: List[str], numerical_features: List[str]):
        super(LSTMModelWithAttention, self).__init__()
        self.config = config
        self.feature_names = feature_names
        self.num_classes = num_classes
        self.numerical_features = numerical_features
        
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size + 1, config.hidden_size, padding_idx=0)
            for feature, vocab_size in vocab_sizes.items()
        })
        
        lstm_input_size = (config.hidden_size * len(self.embeddings)) + (len(feature_names) - len(self.embeddings))
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout_rate,
            bidirectional=True
        )
        
        # Add attention layer
        self.attention = Attention(config.hidden_size)
        
        self.fc = nn.Linear(config.hidden_size * 2, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = self.config.sequence_length
        embeddings_list = []
        numerical_features = []

        for name in self.embeddings.keys():
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx].long()
            embeddings_list.append(self.embeddings[name](feature_data))
 
        for name in self.numerical_features:
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx]
            numerical_features.append(feature_data)

        numerical_features = torch.stack(numerical_features, dim=2)
        output = torch.cat(embeddings_list + [numerical_features], dim=2)

        return output
    
    def get_last_hidden(self, x):
        cat = self._get_embeddings(x)
        
        output, (hidden, _) = self.lstm(cat)
        
        lengths = (x[:, :self.config.sequence_length] != 0).sum(1)
        
        context = self.attention(output, lengths)
        
        return context

    def forward(self, x):
        cat = self._get_embeddings(x)
        
        output, (hidden, _) = self.lstm(cat)
        
        lengths = (x[:, :self.config.sequence_length] != 0).sum(1)
        
        context = self.attention(output, lengths)
        
        out = self.fc(context)
        return self.sigmoid(out)

class LSTMModel(nn.Module):
    def __init__(self, vocab_sizes: List[int], config: ModelConfig, num_classes: int, feature_names: List[str], numerical_features: List[str]):
        super(LSTMModel, self).__init__()
        torch.manual_seed(551)
        self.config = config
        self.feature_names = feature_names
        self.num_classes = num_classes
        self.numerical_features = numerical_features
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size + 1, config.hidden_size, padding_idx=0)
            for feature, vocab_size in vocab_sizes.items()
        })
        lstm_input_size = (config.hidden_size * len(self.embeddings)) + (len(feature_names) - len(self.embeddings))
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout_rate,
            bidirectional=True
        )
        torch.manual_seed(55)
        self.fc = nn.Linear(config.hidden_size * 2, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = self.config.sequence_length
        embeddings_list = []
        numerical_features = []

        for name in self.embeddings.keys():
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx].long()
            embeddings_list.append(self.embeddings[name](feature_data))
 
        for name in self.numerical_features:
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx]
            numerical_features.append(feature_data)

        numerical_features = torch.stack(numerical_features, dim=2)
        output = torch.cat(embeddings_list + [numerical_features], dim=2)

        return output
    
    def get_last_hidden(self, x):
        cat = self._get_embeddings(x)
        
        output, (hidden, _) = self.lstm(cat)
        last_hidden = hidden[-1]
        lengths = (x[:, :self.config.sequence_length] != 0).sum(1)
        last_output = output[torch.arange(output.size(0)), lengths - 1]
        return last_output

    def forward(self, x):

        cat = self._get_embeddings(x)
        
        output, (hidden, _) = self.lstm(cat)
        last_hidden = hidden[-1]
        lengths = (x[:, :self.config.sequence_length] != 0).sum(1)
        last_output = output[torch.arange(output.size(0)), lengths - 1]

        out = self.fc(last_output)
        return self.sigmoid(out)

class LSTMModelA(nn.Module):
    def __init__(self, vocab_sizes: List[int], config: ModelConfig, num_classes: int, feature_names: List[str], numerical_features: List[str]):
        super(LSTMModelA, self).__init__()
        self.config = config
        self.feature_names = feature_names
        self.num_classes = num_classes
        self.numerical_features = numerical_features
        torch.manual_seed(551)
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size + 1, config.hidden_size, padding_idx=0)
            for feature, vocab_size in vocab_sizes.items()
        })
        lstm_input_size = (config.hidden_size * len(self.embeddings)) + (len(feature_names) - len(self.embeddings)) + 3# +2 for numerical features
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout_rate
        )
        self.fc = nn.Linear(config.hidden_size, self.num_classes)
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

    def forward(self, x):
        cat = self._get_embeddings(x)
        
        output, (hidden, _) = self.lstm(cat)
        last_hidden = hidden[-1]  # Shape: (batch_size, hidden_dim)
        lengths = (x[:, :self.config.sequence_length] != 0).sum(1)  # Mask padding
        last_output = output[torch.arange(output.size(0)), lengths - 1]
        out = self.fc(last_output)
        return self.sigmoid(out)