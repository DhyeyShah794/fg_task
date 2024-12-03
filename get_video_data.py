import os
import pandas as pd
import requests
from tqdm import tqdm

csv_file = "assignment_data.csv"
df = pd.read_csv(csv_file)

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading files"):
    performance_score = row['Performance']
    url = row['Video URL']
    filename = os.path.join(output_dir, f"file_{idx}_{performance_score}.mp4")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    except Exception as e:
        print(f"Failed to download {url}: {e}")

print(f"All files saved.")
