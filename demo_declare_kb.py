'''
Dummy example: build an LTN knowledge base from Declare constraints mined on an event log.

    python demo_declare_kb.py

Steps:
    1. generate a small synthetic event log (an emergency-room-like process with planted rules)
    2. mine Declare constraints with pm4py (declare.PM4PyDeclareMiner)
    3. prune and order them (declare.prune_constraints)
    4. build the model input as in create_ngrams (prefixes, flattened feature-major)
    5. construct the knowledge base of LTN predicates (declare.DeclareKB)
    6. use the predicates in LTN formulas and train a dummy neural predicate P with them
'''
import random
import warnings

import ltn
import pandas as pd
import torch

from declare import DeclareKB, PM4PyDeclareMiner, activity_vocab, encode_log, prune_constraints

warnings.filterwarnings("ignore")
random.seed(0)
torch.manual_seed(0)

MAX_PREFIX_LENGTH = 8

### 1. SYNTHETIC EVENT LOG
# Planted behaviour:
#   - every case starts with Registration, (almost always) immediately followed by Triage
#   - critical cases (label 1, e.g. ICU) always get a Lab Test before Antibiotics and are admitted
#   - some non-critical cases get Antibiotics without a Lab Test (they violate Precedence(Lab Test, Antibiotics))
#   - "value" is a numeric event attribute (e.g. a lab value), higher for critical cases
def generate_case(case_id):
    critical = random.random() < 0.4
    trace = ["Registration", "Triage"]
    trace += [random.choice(["Vital Signs", "Lab Test"]) for _ in range(random.randint(0, 2))]
    if critical:
        if "Lab Test" not in trace:
            trace.append("Lab Test")
        trace.append("Antibiotics")
    elif random.random() < 0.3:
        trace.append("Antibiotics")
    trace.append(random.choice(["Vital Signs", "Lab Test"]))
    trace.append("Admission" if critical else random.choice(["Discharge", "Discharge", "Admission"]))
    if random.random() < 0.05:
        trace.remove("Triage")
    return [{"case:concept:name": f"case_{case_id}", "concept:name": act, "label": int(critical),
             "value": min(1.0, max(0.0, random.gauss(0.6 if critical else 0.4, 0.2)))} for act in trace]

log = pd.DataFrame([event for i in range(500) for event in generate_case(i)])
print(f"-- Synthetic log: {log['case:concept:name'].nunique()} cases, {len(log)} events")

# Activity codes as in the preprocessing scripts: pd.Categorical(...).cat.codes + 1, PAD = 0
vocab = activity_vocab(log["concept:name"])
print("Activity codes:", vocab)

### 2. DECLARE DISCOVERY WITH PM4PY
traces, case_ids, labels = encode_log(log, vocab, max_len=MAX_PREFIX_LENGTH)
miner = PM4PyDeclareMiner(min_support=0.1, min_confidence=0.8, prune_redundant=False)
mined = miner.fit(traces, labels, vocab=vocab)
print(f"\n-- {len(mined)} constraints mined with pm4py")

### 3. PRUNING AND ORDERING
constraints = prune_constraints(
    mined,
    max_ltn_gap=0.01,          # pm4py and the LTN predicates must agree on the constraint
    prefix_safe_only=True,     # enforceable on prefixes (violations are permanent)
    exclude_templates=["absence"],
    redundant=True,            # drop constraints implied by stronger ones
    transitive=True,           # drop T(a, c) implied by T(a, b) and T(b, c)
    sort_by=["class_gap", "confidence"],   # constraints behaving differently on the two labels first
    top_k_per_template=2,
    top_k=6,
)
print(f"-- {len(constraints)} constraints after pruning:")
for c in constraints:
    print("  ", c)

### 4. MODEL INPUT (as create_ngrams: prefixes of each case, feature-major, padded to MAX_PREFIX_LENGTH)
feature_names = ["value", "concept:name"]
X, y = [], []
for case_id, group in log.groupby("case:concept:name"):
    group = group.iloc[:MAX_PREFIX_LENGTH]
    for n in range(1, len(group)):
        prefix = group.iloc[:n]
        cols = [prefix[f].tolist() + [0] * (MAX_PREFIX_LENGTH - n) for f in feature_names]
        cols[1] = [vocab[a] if a != 0 else 0 for a in cols[1]]
        X.append(torch.tensor(cols, dtype=torch.float32).flatten())
        y.append(group["label"].iloc[0])
X, y = torch.stack(X).to(ltn.device), torch.tensor(y, dtype=torch.float32).to(ltn.device)
print(f"\n-- Model input: {X.shape[0]} prefixes of shape {tuple(X.shape[1:])}")

### 5. KNOWLEDGE BASE OF LTN PREDICATES
kb = DeclareKB(constraints, feature_names, MAX_PREFIX_LENGTH, vocab=vocab)
print(f"\n-- {kb}")

x_All = ltn.Variable("x_All", X)
print("\nLTN predicates (one per constraint) and their satisfaction on the prefixes:")
for name, predicate in kb.predicates.items():
    truth = predicate(x_All)
    print(f"   {name:<55} {type(predicate).__name__}  sat={truth.value.mean().item():.3f}")

# Template-level predicates with activities as LTN constants
Precedence = kb.template_predicate("precedence")
truth = Precedence(x_All, kb.constant("Lab Test"), kb.constant("Antibiotics"))
print(f"\nPrecedence(x, Lab Test, Antibiotics) as template predicate + constants: sat={truth.value.mean().item():.3f}")

# Crisp satisfaction matrix, e.g. for compliance metrics: (num_prefixes, num_constraints)
print("Satisfaction matrix:", tuple(kb.satisfaction(X).shape))

### 6. LTN FORMULAS WITH THE KNOWLEDGE BASE
Forall, Not, Implies = kb.Forall, kb.Not, kb.Implies
SatAgg = ltn.fuzzy_ops.SatAgg()

P = ltn.Predicate(torch.nn.Sequential(
    torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 1), torch.nn.Sigmoid())).to(ltn.device)
optimizer = torch.optim.Adam(P.parameters(), lr=1e-2)


def knowledge_formulas(x_All):
    # Knowledge satisfaction (no gradient to P, only measures the KB)
    formulas = kb.axioms(x_All, mode="satisfy")
    # Constraint -> label rules: a prefix violating a prefix-safe constraint that always holds on
    # critical cases (conf+ = 1) is not critical
    formulas += kb.filter(lambda c: c.confidence_pos == 1.0).axioms(x_All, P, mode="violation_implies_not")
    # The same rule as a guarded quantifier on one specific constraint
    formulas.append(Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn=kb.cond_fn(kb.constraints[0], satisfied=False)))
    return formulas


def sat_level(x, labels):
    x_P, x_not_P, x_All = ltn.Variable("x_P", x[labels == 1]), ltn.Variable("x_not_P", x[labels == 0]), ltn.Variable("x_All", x)
    formulas = [Forall(x_P, P(x_P)), Forall(x_not_P, Not(P(x_not_P)))] + knowledge_formulas(x_All)
    return SatAgg(*formulas)


print("\n-- Training the dummy neural predicate P with data + Declare knowledge")
for epoch in range(10):
    permutation = torch.randperm(len(X), device=X.device)
    losses = []
    for i in range(0, len(X), 64):
        idx = permutation[i:i + 64]
        optimizer.zero_grad()
        loss = 1 - sat_level(X[idx], y[idx])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    with torch.no_grad():
        accuracy = ((P.model(X).squeeze(-1) > 0.5).float() == y).float().mean().item()
    print(f" epoch {epoch} | loss {sum(losses) / len(losses):.4f} | accuracy {accuracy:.3f}")

with torch.no_grad():
    print(f"\nFinal satisfaction of the whole theory: {sat_level(X, y).item():.3f}")
    predictions = P.model(X).squeeze(-1) > 0.5
    satisfaction = kb.satisfaction(X).bool()
    for k, c in enumerate(kb):
        violating = ~satisfaction[:, k]
        print(f"   {c.name:<45} violated by {violating.sum().item():>4} prefixes, "
              f"predicted critical among them: {predictions[violating].float().mean().item() if violating.any() else 0:.3f}")
