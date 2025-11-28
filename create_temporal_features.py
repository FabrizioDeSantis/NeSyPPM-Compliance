import pandas as pd

df = pd.read_csv("dataset_path/dataset.csv")
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
df = df.sort_values(['case:concept:name', 'time:timestamp'])
df['time_since_previous'] = df.groupby('case:concept:name')['time:timestamp'].diff().dt.total_seconds().fillna(0) / 60
df['elapsed_time'] = df.groupby('case:concept:name')['time:timestamp'].transform(
    lambda x: (x - x.min()).dt.total_seconds() / 60
)
print(df.head())
df.to_csv("dataset_path/result.csv", index=False)