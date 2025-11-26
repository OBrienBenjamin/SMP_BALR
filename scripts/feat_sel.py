import argparse
import sys
import os

import pandas as pd

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from lib.ba_functions import load_data, filter_data, load_feat, format_data, sel_feat_bayes, sel_feat_slda
from lib.export_functions import print_class_stats, export_fs_results

# # # # 

parse = argparse.ArgumentParser()
parse.add_argument("-data_path", type=str, default=None)
parse.add_argument("-method", type=str)
parse.add_argument("-label", type=str)
args = parse.parse_args()

# # # init

# # # ATTENTION! !  Desired labels inserted here # # # 
# # # (for default MSP-Podcast dataset)
VALID = ['A', 'H', 'N', 'S'] if args.label == 'EmoClass' else None

# # # # # # # # # # # # # # # # # # # # # # # #  

print('Loading data . . .')
df = load_data(data_path = args.data_path, step = 'train')
print_class_stats(df, args.label, "Original")

df = filter_data(df, args.label, valid_labels = VALID)
print_class_stats(df, args.label, "Filtered")

print('Loading feature list and formating data')
feature_names = load_feat(df, args.label)
X, Y, label_encoder, _ = format_data(df, args.label, feature_names, class_weight = 0)

print(list(set(Y)))
    
if args.method == 'slda':
    df_slda = sel_feat_slda(
        X, Y, feature_names,
        p_thresh=0.05,
        workers=8,
        batch_size=32
    )
    export_fs_results(df_slda, args.label, args.method)

if args.method == 'bayes':
    df_bayes = sel_feat_bayes(X, Y, feature_names, label = args.label, K = len(label_encoder.classes_), top_features=len(feature_names))
    export_fs_results(df_bayes, args.label, args.method)

print('Voila!')
