import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def split_and_create(dataset):
    # Initialize lists to collect subsets for each category
    accepted_subsets = []
    cancelled_subsets = []
    denied_subsets = []

    # Ensure 'time:timestamp' is datetime type
    dataset['time:timestamp'] = pd.to_datetime(dataset['time:timestamp'], format="mixed", dayfirst=False)

    # Group the dataframe by case ID
    for case_id, case_df in dataset.groupby('case:concept:name'):
        # Iterate through each activity in the case's sequence
        for pos in range(len(case_df)):
            current_activity = case_df.iloc[pos]['concept:name']
            # Check if the current activity is one of the target activities
            if current_activity in ['O_Accepted', 'A_Cancelled', 'A_Denied']:
                # Extract the subset up to BUT NOT INCLUDING the current activity
                subset = case_df.iloc[:pos].copy()
                
                # Calculate time difference between the target activity (excluded) and its previous activity
                if pos >= 1:
                    # Get target activity's timestamp (pos) and previous activity's timestamp (pos-1)
                    current_time = case_df.iloc[pos]['time:timestamp']
                    previous_time = case_df.iloc[pos-1]['time:timestamp']
                    time_diff = current_time - previous_time
                else:
                    # Target is the first activity: no previous activity to include
                    time_diff = pd.Timedelta(0)
                
                # Add time difference and label columns to the subset (if subset is not empty)
                if not subset.empty:
                    subset['time_diff'] = time_diff
                    subset['label_wt'] = int(time_diff < pd.Timedelta(days=1))
                    
                    # Append to the corresponding list based on the activity type
                    if current_activity == 'O_Accepted':
                        accepted_subsets.append(subset)
                        subset["label"] = 1
                    elif current_activity == 'A_Cancelled':
                        cancelled_subsets.append(subset)
                        subset["label"] = 0
                    elif current_activity == 'A_Denied':
                        denied_subsets.append(subset)
                        subset["label"] = 0
                
                # Break the loop as we only need the first occurrence
                break

    # Concatenate all subsets into respective DataFrames, handling empty cases
    accepted_df = pd.concat(accepted_subsets, ignore_index=True) if accepted_subsets else pd.DataFrame()
    cancelled_df = pd.concat(cancelled_subsets, ignore_index=True) if cancelled_subsets else pd.DataFrame()
    denied_df = pd.concat(denied_subsets, ignore_index=True) if denied_subsets else pd.DataFrame()
    # accepted_df.to_csv('accepted.csv', index=False)
    # cancelled_df.to_csv('cancelled.csv', index=False)
    # denied_df.to_csv('refused.csv', index=False)
    
    df = pd.concat([accepted_df, cancelled_df, denied_df], ignore_index=True)
    df.to_csv("data_processed/BPIC2017_filtered.csv", index=False)

def pad_lists(lists, num_features, max_length):
    return [tuple(lst[i] + [0] * (max_length - len(lst[i])) for i in range(num_features)) for lst in lists]

def print_dataset_info(training_list, validation_list, test_list):
    print("-- DATASET INFO")
    print(f"Training set size: {len(training_list)}")
    print(f"Validation set size: {len(validation_list)}")
    print(f"Test set size: {len(test_list)}")
    print("--- Label distribution")
    print("--- Training set")
    counts = Counter(training_list[1])
    print(counts)
    print("--- Validation set")
    counts = Counter(validation_list[1])
    print(counts)
    print("--- Test set")
    counts = Counter(test_list[1])
    print(counts)

def plot_compliance(compliance_dict):
    # Sort the lengths for proper ordering in the plot
    lengths = sorted(compliance_dict.keys())
    compliance_scores = [compliance_dict[length] for length in lengths]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lengths, compliance_scores, marker='o', linestyle='-', color='b')
    
    # Add labels and title
    ax.set_xlabel('Case Length', fontsize=12)
    ax.set_ylabel('Compliance Score', fontsize=12)
    ax.set_title('Compliance by Case Length', fontsize=14)
    
    # Set y-axis limits and grid
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()