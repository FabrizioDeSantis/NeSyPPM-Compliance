'''
Configurations, losses and training loop shared by all the experiments.

    vanilla    backbone trained with binary cross-entropy (BCE)
    feat_eng   backbone with the rule truth values as extra inputs, BCE
    semantic   backbone, BCE + semantic loss on the rules
    ltn        LTN with the data axioms only:  Forall x+: P(x+),  Forall x-: Not(P(x-))
    A, B, C and their combinations (AB, AC, BC, ABC), LTN with knowledge:
        A  the truth values of the logical rules (nesy/knowledge.py) are extra inputs of P
        B  the logical rules are axioms on P
        C  the Declare (control-flow) constraints of nesy/declare_knowledge.py, never connected to the label:
           with --declare_in_p (default) their truth values are extra inputs of P, i.e. P(x, C(x));
           with --no-declare_in_p they stay out of P and are axioms Forall x: C(x) of the knowledge base
           (they involve no learnable predicate, so they do not change the gradients of P)
'''
import copy
import os
import random
import statistics
from dataclasses import dataclass

import ltn
import numpy as np
import torch
import torch.nn.functional as F

from model.lstm import LSTMModel
from model.semantic_loss import semantic_loss
from model.transformer import EventTransformer

CONFIGS = ["vanilla", "feat_eng", "semantic", "ltn", "A", "B", "C", "AB", "AC", "BC", "ABC"]


@dataclass(frozen=True)
class Configuration:
    name: str
    loss: str                      # "bce", "semantic" or "ltn"
    rule_features: bool = False    # feat_eng / A
    rule_axioms: bool = False      # B
    declare_features: bool = False  # C, inputs of P (--declare_in_p)
    declare_axioms: bool = False    # C, axioms Forall x: C(x) (--no-declare_in_p)


def parse_configuration(name, declare_in_p=True):
    if name == "vanilla":
        return Configuration(name, "bce")
    if name == "feat_eng":
        return Configuration(name, "bce", rule_features=True)
    if name == "semantic":
        return Configuration(name, "semantic")
    if name == "ltn":
        return Configuration(name, "ltn")
    if name in CONFIGS:
        return Configuration(name, "ltn", rule_features="A" in name, rule_axioms="B" in name,
                             declare_features="C" in name and declare_in_p,
                             declare_axioms="C" in name and not declare_in_p)
    raise ValueError(f"Unknown configuration {name}, expected one of {CONFIGS}")


def set_seed(seed):
    '''Seed every source of randomness, and make cuDNN deterministic.'''
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class ModelInput:
    '''
    Input of the model: the prefixes x followed by the truth values of the logical rules (feat_eng, A) and of the
    Declare constraints (C), which the model reads as extra features (see model/inputs.EventEncoder).
    '''

    def __init__(self, config, knowledge, declare=None):
        self.knowledge = knowledge if config.rule_features else None
        self.declare = declare if config.declare_features else None

    @property
    def num_features(self):
        return ((self.knowledge.num_features if self.knowledge else 0) +
                (self.declare.num_features if self.declare else 0))

    def __call__(self, x):
        parts = [x]
        if self.knowledge is not None:
            parts.append(self.knowledge.truth(x).float())
        if self.declare is not None:
            parts.append(self.declare.truth(x).float())
        return torch.cat(parts, dim=1) if len(parts) > 1 else x


def build_model(args, data, num_rule_features=0):
    layout = data.layout
    if args.model == "lstm":
        return LSTMModel(list(layout.feature_names), data.vocab_sizes, layout.seq_len, hidden_size=args.hidden_size,
                         num_layers=args.num_layers, dropout=args.dropout, num_rule_features=num_rule_features)
    return EventTransformer(list(layout.feature_names), data.vocab_sizes, layout.seq_len, model_dim=args.model_dim,
                            num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                            num_rule_features=num_rule_features)


### LOSSES: functions (x, y) -> scalar loss

def bce_loss(model, model_input):
    def loss(x, y):
        return F.binary_cross_entropy(model(model_input(x)).squeeze(1), y)
    return loss


def semantic_rule_loss(model, knowledge, lambda_sem):
    def loss(x, y):
        truth = knowledge.truth(x, knowledge.rules)
        labels = knowledge.labels(knowledge.rules, device=x.device)
        return semantic_loss(model(x).squeeze(1), y, truth[:, labels == 1], truth[:, labels == 0], lambda_sem)
    return loss


class LTNLoss:
    '''1 - SatAgg of the axioms of one LTN configuration, grounded on a batch.'''
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    SatAgg = ltn.fuzzy_ops.SatAgg()

    def __init__(self, P, model_input, knowledge, config, declare=None):
        self.P = P
        self.model_input = model_input
        self.knowledge = knowledge
        self.config = config
        self.declare = declare if config.declare_axioms else None

    def axioms(self, x, y):
        P, Forall, Not = self.P, self.Forall, self.Not
        x_in = self.model_input(x)
        axioms = []
        # Data: P holds on the positive prefixes and not on the negative ones
        if (y == 1).any():
            x_pos = ltn.Variable("x_pos", x_in[y == 1])
            axioms.append(Forall(x_pos, P(x_pos)))
        if (y == 0).any():
            x_neg = ltn.Variable("x_neg", x_in[y == 0])
            axioms.append(Forall(x_neg, Not(P(x_neg))))
        if self.config.rule_axioms:
            # B: Forall x with condition(x): P(x), or Not(P(x)) for the rules implying label 0.
            # A rule firing on no prefix of the batch has no grounding and is skipped.
            x_all = ltn.Variable("x_all", x_in)
            P_all = P(x_all)
            truth = self.knowledge.truth(x, self.knowledge.rules)
            for k, rule in enumerate(self.knowledge.rules):
                if truth[:, k].any():
                    head = P_all if rule.label == 1 else Not(P_all)
                    axioms.append(Forall(x_all, head, cond_vars=[x_all], cond_fn=lambda _, fired=truth[:, k]: fired))
        if self.declare is not None:
            # C outside P: Forall x: C(x) for each Declare constraint
            axioms.extend(self.declare.axioms(ltn.Variable("x_trace", x)))
        return axioms

    def __call__(self, x, y):
        return 1 - self.SatAgg(*self.axioms(x, y))


### TRAINING LOOP

def fit(model, loss_fn, optimizer, train_loader, validate, epochs, patience, device, name):
    '''
    Minimize loss_fn(x, y) for at most `epochs` epochs. After each epoch the model is scored with
    validate() (validation macro-F1); the best parameters are restored at the end, and training stops
    after `patience` epochs without improvement.
    '''
    best_score, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(x, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        score = validate()
        improved = score > best_score
        print(f"   [{name}] epoch {epoch}/{epochs} | train loss {statistics.mean(losses):.4f} | "
              f"val F1 {score:.4f}{' *' if improved else ''}")
        if improved:
            best_score, best_state, bad_epochs = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"   [{name}] early stopping: no improvement for {patience} epochs")
                break
    model.load_state_dict(best_state)
    return best_score
