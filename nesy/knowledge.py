'''
Domain knowledge of each dataset, as crisp rules on the model input.

A rule says: if condition(x) holds for a prefix x, its label should be `label`. Each rule is defined once
here, on named features, and used in the same way by every configuration:
    feat_eng, A   the rule truth values are appended to the model input (nesy/training.ModelInput)
    semantic      semantic loss towards the label of the fired rules
    B             LTN axiom  Forall x with condition(x): P(x)   (Not(P(x)) when label = 0)
    evaluation    compliance: among the test prefixes where a rule fires and the true label agrees with it,
                  the fraction whose prediction agrees too
Rules with constraint_only=True only constrain the predictions (semantic loss, B axioms): they are neither
input features nor counted in the compliance.
'''
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import torch


@dataclass(frozen=True)
class Rule:
    name: str
    label: int
    condition: Callable  # (B, width) model input -> (B,) bool
    constraint_only: bool = False
    formula: str = ""    # first-order formula, as in knowledge_base.txt


class Knowledge:
    def __init__(self, rules):
        self.rules = list(rules)
        self.feature_rules = [r for r in self.rules if not r.constraint_only]

    @property
    def num_features(self):
        '''Number of rule truth values appended to the model input (feat_eng, A).'''
        return len(self.feature_rules)

    def truth(self, x, rules=None):
        '''(B, K) bool: truth of the antecedent (condition) of each rule (default: feature_rules) on x.'''
        rules = self.feature_rules if rules is None else rules
        if not rules:
            return torch.zeros(len(x), 0, dtype=torch.bool, device=x.device)
        return torch.stack([r.condition(x) for r in rules], dim=1)

    def labels(self, rules=None, device=None):
        rules = self.feature_rules if rules is None else rules
        return torch.tensor([r.label for r in rules], device=device)

    def print_summary(self, x, y):
        '''How often each rule fires on (x, y) and how often the label agrees with it.'''
        print("-- Knowledge (on the training prefixes)")
        truth = self.truth(x, self.rules)
        for k, rule in enumerate(self.rules):
            fired = truth[:, k]
            n = int(fired.sum())
            agree = int((y[fired] == rule.label).sum())
            extra = "  [constraint only]" if rule.constraint_only else ""
            precision = f"{agree / n:.3f}" if n else "n/a"
            print(f"   {rule.name}: fires on {n} prefixes, label agrees on {precision}{extra}")


### HELPERS

def scaled(scaler, value):
    '''`value` in the units of a feature normalized by a fitted MinMaxScaler.'''
    return float(value * scaler.scale_[0] + scaler.min_[0])


def unscaled(scaler, values):
    '''Inverse of the MinMaxScaler normalization (tensor of normalized values -> original units).'''
    return (values - float(scaler.min_[0])) / float(scaler.scale_[0])


def category_codes(raw, column):
    '''Value -> code, as assigned by the preprocessing (pd.Categorical(column).cat.codes + 1).'''
    return {value: i + 1 for i, value in enumerate(pd.Categorical(raw[column]).categories)}


def occurs(acts, a):
    '''(B,) activity a occurs in the prefix.'''
    return (acts == a).any(dim=1)


def occurs_before(acts, a, b):
    '''(B,) some occurrence of activity a precedes some occurrence of activity b.'''
    is_a = (acts == a).long()
    a_seen_before = (is_a.cumsum(dim=1) - is_a) > 0
    return (a_seen_before & (acts == b)).any(dim=1)


def occur_within(acts, times, a, b, max_gap):
    '''(B,) some occurrence of a and some occurrence of b are at most max_gap apart (in either order).'''
    pairs = (acts == a).unsqueeze(2) & (acts == b).unsqueeze(1)
    close = (times.unsqueeze(2) - times.unsqueeze(1)).abs() <= max_gap
    return (pairs & close).any(dim=2).any(dim=1)


### RULES OF EACH DATASET
# Case attributes (e.g. the requested amount) have the same value at every event: they are read at the first one.

def bpi12_rules(data):
    layout, amount = data.layout, data.scalers["case:AMOUNT_REQ"]
    resource = category_codes(data.raw, "org:resource")
    res_11169, res_10910 = resource["11169"], resource["10910"]
    lt_10k, gt_40k, le_60k = scaled(amount, 10_000), scaled(amount, 40_000), scaled(amount, 60_000)

    def requested_amount(x):
        return layout.get(x, "case:AMOUNT_REQ")[:, 0]

    def resources(x):
        return layout.get(x, "org:resource")

    # Label 1: the application is accepted
    return [
        Rule("R1: amount < 10k -> not accepted", 0, lambda x: requested_amount(x) < lt_10k,
             formula="Forall x (RequestedAmount(x) < 10000 -> Not(Accepted(x)))"),
        Rule("R2: 40k < amount <= 60k -> not accepted", 0,
             lambda x: (requested_amount(x) > gt_40k) & (requested_amount(x) <= le_60k),
             formula="Forall x (RequestedAmount(x) > 40000 AND RequestedAmount(x) <= 60000 -> Not(Accepted(x)))"),
        Rule("R3: resource 11169 or 10910 involved -> not accepted", 0,
             lambda x: occurs(resources(x), res_11169) | occurs(resources(x), res_10910),
             formula="Forall x (HasResource(x, 10910) OR HasResource(x, 11169) -> Not(Accepted(x)))"),
    ]


def bpi17_rules(data):
    layout, amount = data.layout, data.scalers["case:RequestedAmount"]
    create_offer = category_codes(data.raw, "concept:name")["O_Create Offer"]
    loan_takeover = category_codes(data.raw, "case:LoanGoal")["Existing loan takeover"]
    limit_20k = scaled(amount, 20_000)

    def requested_amount(x):
        return layout.get(x, "case:RequestedAmount")[:, 0]

    def offer_created(x):
        return occurs(layout.get(x, "concept:name"), create_offer)

    def has_credit_score(x):
        return (layout.get(x, "CreditScore") != 0).any(dim=1)

    # Label 1: the offer is accepted
    return [
        Rule("R1: amount < 20k, offer created with credit score -> accepted", 1,
             lambda x: (requested_amount(x) < limit_20k) & offer_created(x) & has_credit_score(x),
             formula="Forall x (RequestedAmount(x) < 20000 AND HasActivity(x, O_Create Offer) AND CreditScore(x) > 0 "
                     "-> Accepted(x))"),
        Rule("R2: offer created without credit score -> not accepted", 0,
             lambda x: offer_created(x) & ~has_credit_score(x),
             formula="Forall x (HasActivity(x, O_Create Offer) AND Not(CreditScore(x) > 0) -> Not(Accepted(x)))"),
        Rule("R3: amount > 20k, existing loan takeover -> not accepted", 0,
             lambda x: (requested_amount(x) > limit_20k) & occurs(layout.get(x, "case:LoanGoal"), loan_takeover),
             formula="Forall x (RequestedAmount(x) > 20000 AND LoanGoal(x) = \"Existing loan takeover\" "
                     "-> Not(Accepted(x)))"),
    ]


def sepsis_rules(data):
    layout, scalers = data.layout, data.scalers
    activity = category_codes(data.raw, "concept:name")
    crp, antibiotics, er_triage = activity["CRP"], activity["IV Antibiotics"], activity["ER Triage"]
    lactic_2, crp_100 = scaled(scalers["LacticAcid"], 2), scaled(scalers["CRP"], 100)
    gap_120 = 120 * float(scalers["elapsed_time"].scale_[0])  # a difference of 120 in elapsed_time units

    def acts(x):
        return layout.get(x, "concept:name")

    def flag(x, name):
        return layout.get(x, name).eq(1).any(dim=1)

    # Label 1: the patient is admitted to the ICU
    return [
        Rule("R1: lactic acid > 2 -> ICU", 1, lambda x: (layout.get(x, "LacticAcid") > lactic_2).any(dim=1),
             formula="Forall x (LacticAcid(x) > 2 -> ICU(x))"),
        Rule("R2: infection suspected, SIRS tachypnea and heart rate -> ICU", 1,
             lambda x: flag(x, "InfectionSuspected") & flag(x, "SIRSCritTachypnea") & flag(x, "SIRSCritHeartRate"),
             formula="Forall x (SuspectedInfection(x) AND Tachypnea(x) AND CriticalHeartRate(x) -> ICU(x))"),
        Rule("R3: CRP before IV antibiotics, CRP > 100 -> ICU", 1,
             lambda x: occurs_before(acts(x), crp, antibiotics) & (layout.get(x, "CRP") > crp_100).any(dim=1),
             formula="Forall x (Precedes(x, CRP, IV Antibiotics) AND CRPValue(x) > 100 -> ICU(x))"),
        Rule("R7: ER triage and IV antibiotics within 120 -> no ICU", 0,
             lambda x: occur_within(acts(x), layout.get(x, "elapsed_time"), er_triage, antibiotics, gap_120),
             constraint_only=True,
             formula="Forall x (|ElapsedTime(x, ER Triage) - ElapsedTime(x, IV Antibiotics)| <= 120 -> Not(ICU(x)))"),
    ]


def traffic_rules(data):
    layout, scalers = data.layout, data.scalers
    activity = category_codes(data.raw, "concept:name")
    add_penalty, payment = activity["Add penalty"], activity["Payment"]
    amount_400 = scaled(scalers["amount"], 400)

    def acts(x):
        return layout.get(x, "concept:name")

    def paid_less_than_fine(x):
        # amount and paymentAmount have different scalers: compare them in the original units
        fine = unscaled(scalers["amount"], layout.get(x, "amount")[:, 0])
        paid = unscaled(scalers["paymentAmount"], layout.get(x, "paymentAmount")).max(dim=1).values
        return paid < fine

    # Label 1: the fine is sent for credit collection
    return [
        Rule("R1: paid less than the fine -> credit collection", 1,
             lambda x: occurs(acts(x), payment) & paid_less_than_fine(x),
             formula="Forall x (HasActivity(x, Payment) AND PaymentAmount(x) < FineAmount(x) -> CreditCollection(x))"),
        Rule("R2: penalty added -> credit collection", 1, lambda x: occurs(acts(x), add_penalty),
             formula="Forall x (HasActivity(x, Add penalty) -> CreditCollection(x))"),
        Rule("R3: amount > 400 -> credit collection", 1, lambda x: layout.get(x, "amount")[:, 0] > amount_400,
             formula="Forall x (FineAmount(x) > 400 -> CreditCollection(x))"),
    ]


RULES = {"bpi12": bpi12_rules, "bpi17": bpi17_rules, "sepsis": sepsis_rules, "traffic": traffic_rules}


def build_knowledge(data):
    return Knowledge(RULES[data.name](data))
