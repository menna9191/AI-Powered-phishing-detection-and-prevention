import os
import glob
import re
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import ast


# Extract the initial number from the file
def extract_start(filename):
    basename = os.path.basename(filename)
    match = re.search(r'_(\d+)-\d+\.csv', basename)
    if match:
        return int(match.group(1))
    return 0


# Change unique_dns_tld_set into set
def to_set(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return set()
    elif isinstance(x, (set, list)):
        return set(x)
    else:
        return set()


# Find the files that match the corresponding pattern
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../..', 'data')
pattern = os.path.join(data_dir, "final_dataset_with_features_*-[0-9]*.csv")
files = glob.glob(pattern)

# Sort the files according to the initial number and combine them
files_sorted = sorted(files, key=extract_start)
print("Files sorted in order:", files_sorted)
df_list = []
for file in files_sorted:
    print(f"Reading file: {file}")
    df_temp = pd.read_csv(file)
    df_list.append(df_temp)
combined_df = pd.concat(df_list, ignore_index=True)
print(f"Total records after merge: {len(combined_df)}")

# Apply one-hot encoding for "protocol" column
protocol_dummies = pd.get_dummies(combined_df['protocol'],
                                  prefix='protocol')
combined_df = pd.concat([combined_df, protocol_dummies], axis=1)
combined_df.drop(columns=['protocol'], inplace=True)

# Apply multi-hot encoding for unique_dns_tld_set column
combined_df['unique_dns_tld_set'] = combined_df['unique_dns_tld_set']\
                                    .apply(to_set)
mlb = MultiLabelBinarizer()
tld_encoded = mlb.fit_transform(combined_df['unique_dns_tld_set'])
tld_encoded_df = pd.DataFrame(tld_encoded, columns=[f'tld_{cls}'
                              for cls in mlb.classes_])
combined_df = pd.concat([combined_df, tld_encoded_df], axis=1)
combined_df.drop(columns=['unique_dns_tld_set'], inplace=True)

# When the number of 1s in column tld_XXX is less than the
# threshold, the column is dropped
threshold = 50
tld_columns = [f'tld_{cls}' for cls in mlb.classes_]
drop_cols = [col for col in tld_columns if
             combined_df[col].sum() < threshold]
combined_df.drop(columns=drop_cols, inplace=True)

# Save the final features
output_file = os.path.join(data_dir, "final_dataset_with_features.csv")
combined_df.to_csv(output_file, index=False)
print(f"Final dataset saved at: {output_file}")
