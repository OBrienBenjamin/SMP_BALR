import argparse
import sys
import os

import pandas as pd

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from lib.ba_functions import collect_feats

# # # # 

parse = argparse.ArgumentParser()
parse.add_argument("-include", nargs='+', type=str)
parse.add_argument("-exclude", nargs='+', type=str)
parse.add_argument("-output", nargs='+', type=str)
args = parse.parse_args()

# # # init
features_dir = os.path.join(project_root, "features")

# # # # # # # # # # # # # # # # # # # # # # # #  
common_feats = collect_feats(args.include, args.exclude)
print(f'Number of common features : {len(common_feats)}')

for output_name in args.output:
    output_path = os.path.join(features_dir, f'{output_name}_features.csv')
    pd.DataFrame(common_feats, columns=['Feature']).to_csv(output_path, index=False)
    
print('Voila!')
