import json
import pandas as pd
import os


# 1. Read phishing_data.txt and extract phishing urls
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../..', 'data')
phishing_file = os.path.join(data_dir, 'phishing_data.txt')
phishing_df = pd.read_csv(phishing_file, comment='#', header=0,
                          names=['id', 'dateadded', 'url',
                                 'url_status', 'last_online',
                                 'threat', 'tags',
                                 'urlhaus_link', 'reporter'])
malicious_urls = phishing_df['url'].dropna().tolist()
malicious_df = pd.DataFrame({'url': malicious_urls, 'label': 1})
print(f"Extract {len(malicious_df)} phishing urls from "
      f"phishing_data.txt")


# 2. Read cdx-00000
cdx_file = os.path.join(data_dir, 'cdx-00000')
cdx_buffer = []
start_line = 1
end_line = 200000

# Open cdx-00000 file
with open(cdx_file, "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        # Skip the lines before the start line
        if line_number < start_line:
            continue
        if line_number > end_line:
            break
        if not line.strip():
            continue

        # Split the data into index, timestamp and json data
        # Skip the line if it can't be divided correctly
        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue
        _, _, json_data = parts

        # Parse the json data
        try:
            json_obj = json.loads(json_data)
        except json.JSONDecodeError:
            continue

        # Label = 1 if the url is in phishing urls
        url = json_obj.get("url")
        label = 1 if url in malicious_urls else 0

        # Save the url and label
        cdx_buffer.append({"url": url, "label": label})

cdx_df = pd.DataFrame(cdx_buffer)
print(f"Scan {len(cdx_df)} records in cdx-00000.")


# 3. Extract the same amount of valid (0) and invalid (1) urls
# Separate the valid and invalid urls in cdx-00000
mal_df = cdx_df[cdx_df["label"] == 1]
leg_df = cdx_df[cdx_df["label"] == 0]
print(f"{len(leg_df)} valid urls, {len(mal_df)} phishing urls "
      f"in cdx-00000.")

# Get the same amount of valid urls as phishing_data.txt
legal_df = leg_df.sample(n=len(malicious_urls), random_state=42)

# Combine the two datasets and shuffle the order
combined_dataset = pd.concat([legal_df, malicious_df]).\
    sample(frac=1, random_state=42).reset_index(drop=True)


# 4. Save the constructed dataset
if not combined_dataset.empty:
    dataset_output = os.path.join(data_dir, f'final_dataset.csv')
    combined_dataset.to_csv(dataset_output, index=False)
    print(f"Successfully constructed the data.\n"
          f"{len(combined_dataset)} records in total.\n"
          f"Saved in {dataset_output}.")
else:
    print("No valid data was found. The final dataset was"
          " not successfully generated.")
