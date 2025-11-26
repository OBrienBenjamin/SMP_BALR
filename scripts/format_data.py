import pandas as pd
import argparse
import sys
import os

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from lib.ba_functions import extract_balr

parse = argparse.ArgumentParser()
parse.add_argument("-data_path", type=str)
parse.add_argument("-label_path", type=str)
parse.add_argument("-name", type=str)
parse.add_argument("-split", type=int, default=0)
args = parse.parse_args()

# # # # # # # # # # # # # # # # 

print('Loading label data . . .')
df = pd.read_csv(args.label_path)

print('Extracting BALRv2 vectors. . .')
ba_features = extract_balr(args.data_path)

print('Concatenating dataframes, exporting')
merged = pd.merge(df, ba_features, on="FileName", how="inner")

if args.split:
    print('Splitting data into train, test dataframes')
    train_df = merged[merged["Group"] == "Train"].reset_index(drop=True)
    test_df  = merged[merged["Group"] == "Test"].reset_index(drop=True)

    train_df.to_csv(os.path.join(project_root, 'data', f'{args.name}_train_data.csv'), index=False)
    test_df.to_csv(os.path.join(project_root, 'data', f'{args.name}_test_data.csv'), index=False)
else:
    merged.to_csv(os.path.join(project_root, 'data', f'{args.name}_data.csv'), index=False)
    
print('Volia')