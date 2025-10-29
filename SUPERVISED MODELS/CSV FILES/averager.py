import os
import glob
import pandas as pd

# Absolute path of the current folder
current_folder = os.getcwd()
print(f"Current working directory: {current_folder}")

# See all files (not just CSV) — this is your confirmation step
all_files = os.listdir(current_folder)
print("\nAll files in the folder:")
for file in all_files:
    print("  ", file)

# Match CSVs (case-insensitive, safe pattern)
csv_files = glob.glob("*.[cC][sS][vV]")
print(f"\nCSV files found: {csv_files}")

# Try to read and average
dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not dfs:
    print("No valid CSV files loaded. Exiting.")
    exit()

all_data = pd.concat(dfs)
averaged_results = all_data.groupby('Model').mean().reset_index()
print("\nAveraged results:")
print(averaged_results)

averaged_results.to_csv("averaged_model_results.csv", index=False)
print("\nSaved as averaged_model_results.csv")
