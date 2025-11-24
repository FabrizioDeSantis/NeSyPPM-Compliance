from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def create_test_set(data, seed, ratio=0.2):
    random.seed(seed)
    grouped = data.groupby("case:concept:name")
    unique_groups = list(grouped.groups.keys())  # Get the unique group IDs
    labels = data.groupby("case:concept:name")["label"].first().to_list()

    len_test_set = int(len(unique_groups) * ratio)

    labels_r1 = data.groupby("case:concept:name")["rule_1"].first().to_list()
    labels_r2 = data.groupby("case:concept:name")["rule_2"].first().to_list()
    labels_r3 = data.groupby("case:concept:name")["rule_3"].first().to_list()

    filtered_values_r1 = [v for v, l, r in zip(unique_groups, labels, labels_r1) if (r == 1 and l == 1)]
    filtered_values_r2 = [v for v, l, r in zip(unique_groups, labels, labels_r2) if (r == 1 and l == 1)]
    filtered_values_r3 = [v for v, l, r in zip(unique_groups, labels, labels_r3) if (r == 1 and l == 1)]

    filtered_values_nor = [v for v, l, r1, r2, r3 in zip(unique_groups, labels, labels_r1, labels_r2, labels_r3) if (r1 == 0 and r2 == 0 and r3 == 0 and l == 0)]

    filtered_values = filtered_values_r1 + filtered_values_r2 + filtered_values_r3 + filtered_values_nor
    compliant_ids = list(set(filtered_values_r1 + filtered_values_r2 + filtered_values_r3))
    filtered_values = list(set(filtered_values))

    if len(filtered_values) > len_test_set:
        test_ids = random.sample(filtered_values, len_test_set)
    else:
        test_ids = filtered_values

    training_ids = [x for x in unique_groups if x not in test_ids]
    compliant_training_ids = [x for x in training_ids if x in compliant_ids]
    compliant_test_ids = [x for x in test_ids if x in compliant_ids]
    print("Number of compliant traces in training set: ", len(compliant_training_ids))
    print("Number of compliant traces in test set: ", len(compliant_test_ids))

    return training_ids, test_ids

def create_ngrams(data, train_ids, val_ids, test_ids, window_size=13):

    ngrams_test = []
    ngrams_training = []
    ngrams_val = []
    labels_training = []
    labels_val = []
    labels_test = []

    training_data = data[data["case:concept:name"].isin(train_ids)]
    val_data = data[data["case:concept:name"].isin(val_ids)]
    test_data = data[data["case:concept:name"].isin(test_ids)]

    max_len = 13

    for id_value, group in training_data.groupby('case:concept:name'):

        group = group.reset_index(drop=True)
                
        label = int(group['label'].dropna().iloc[0])
        if label == 0:
            idx = group[group['concept:name_str'].str.contains(r'\bRelease\b', na=False)].index
            if not idx.empty:
                group = group.iloc[:idx[0]]

        if len(group) > window_size:
            group = group.iloc[:window_size]

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "lifecycle:transition", "concept:name_str", "rule_1", "rule_2", "rule_3"])

        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            labels_training.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_training.append(cols)

    for id_value, group in val_data.groupby('case:concept:name'):

        group = group.reset_index(drop=True)
                
        label = int(group['label'].dropna().iloc[0])
        if label == 0:
            idx = group[group['concept:name_str'].str.contains(r'\bRelease\b', na=False)].index
            if not idx.empty:
                group = group.iloc[:idx[0]]

        if len(group) > window_size:
            group = group.iloc[:window_size]

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "lifecycle:transition", "concept:name_str", "rule_1", "rule_2", "rule_3"])

        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            labels_val.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_val.append(cols)

    for id_value, group in test_data.groupby('case:concept:name'):

        group = group.reset_index(drop=True)
        
        label = int(group['label'].dropna().iloc[0])
        if label == 0:
            idx = group[group['concept:name_str'].str.contains(r'\bRelease\b', na=False)].index
            if not idx.empty:
                group = group.iloc[:idx[0]]

        if len(group) > window_size:
            group = group.iloc[:window_size]

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "lifecycle:transition", "concept:name_str", "rule_1", "rule_2", "rule_3"])
        
        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            labels_test.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_test.append(cols)

    return ngrams_training, labels_training, ngrams_val, labels_val, ngrams_test, labels_test, feature_names

def preprocess_eventlog(data, seed):

    vocab_sizes = {}
    admissions = data[data["concept:name"] == "ER Registration"]
    labels = admissions["label"].to_list()
    admission_ids = admissions["case:concept:name"].to_list()
    print(len(admission_ids))
    print("Number of patients: ", len(labels))

    train_ids, test_ids = create_test_set(data, seed)
    data_train = data[data["case:concept:name"].isin(train_ids)]
    train_ids, val_ids = create_test_set(data_train, seed, ratio=0.2)

    print("Number of patients in train set: ", len(train_ids))
    print("Number of patients in val set: ", len(val_ids))
    print("Number of patients in test set: ", len(test_ids))

    scaler_la = MinMaxScaler()
    scaler_le = MinMaxScaler()
    scaler_crp = MinMaxScaler()
    scaler_age = MinMaxScaler()
    
    labels = data.groupby("case:concept:name")["label"].first().reset_index()
    print(data.columns)
    data = data.drop(columns=["org:group"])
    for col_name in data.columns.tolist():
        if col_name not in ["case:concept:name", "label", "concept:name", "org:group", "time:timestamp", "Diagnose", "lifecycle:transition", "LacticAcid", "CRP", "Leucocytes"]:
            data[col_name] = data[col_name].fillna(0).astype(int)

    data["concept:name_str"] = data["concept:name"]
    data["concept:name"] = pd.Categorical(data["concept:name"])
    print("CRP code: ", data["concept:name"].cat.categories.get_loc("CRP") + 1)
    print("IV ATB code: ", data["concept:name"].cat.categories.get_loc("IV Antibiotics") + 1)
    print("ER Triage code: ", data["concept:name"].cat.categories.get_loc("ER Triage") + 1)
    print("ER Sepsis Triage code: ", data["concept:name"].cat.categories.get_loc("ER Sepsis Triage") + 1)
    data["concept:name"] = data["concept:name"].cat.codes + 1
    vocab_sizes["concept:name"] = data["concept:name"].max()

    data["Diagnose"] = pd.Categorical(data["Diagnose"]) 
    data["Diagnose"] = data["Diagnose"].cat.codes + 1
    vocab_sizes["Diagnose"] = data["Diagnose"].max()

    # Numerical values
    data["LacticAcid"] = data["LacticAcid"].fillna(0)
    data["LacticAcid"] = scaler_la.fit_transform(data[["LacticAcid"]])
    data["CRP"] = data["CRP"].fillna(0)
    data["CRP"] = scaler_crp.fit_transform(data[["CRP"]])
    data["Leucocytes"] = data["Leucocytes"].fillna(0)
    data["Leucocytes"] = scaler_le.fit_transform(data[["Leucocytes"]])
    data["Age"] = data["Age"].fillna(0)
    data["Age"] = scaler_age.fit_transform(data[["Age"]])

    scalers = {
        "LacticAcid": scaler_la,
        "CRP": scaler_crp,
        "Leucocytes": scaler_le,
        "Age": scaler_age
    }

    print(data.columns)

    return create_ngrams(data, train_ids, val_ids, test_ids), vocab_sizes, scalers