import argparse
import sys
import os

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from lib.ba_functions import load_data, filter_data, load_feat, format_data, extract_shapley_values
from lib.export_functions import print_class_stats, load_encoder

parse = argparse.ArgumentParser()
parse.add_argument("-method", type=str)
parse.add_argument("-label", type=str)
parse.add_argument("-model", type=str)
parse.add_argument("-feats", type=str, default=None)
parse.add_argument("-data_path", type=str, default=None)
parse.add_argument("-ba", nargs='+', type=str)
parse.add_argument("-N", type = int)
args = parse.parse_args()

# # # init

# # # ATTENTION! !  Desired labels inserted here # # # 
VALID = ['A', 'H', 'N', 'S'] if args.label == 'EmoClass' else None

# # # # # # # # # # # # # # # # 
model_name = args.model

print('Loading data . . .')
df = load_data(data_path = args.data_path, step = 'test')
print_class_stats(df, args.label, "Original")

print('Formatting training data . . .')
df = filter_data(df, args.label, valid_labels = VALID, filt_speakers = 0, filt_samples = 0)
print_class_stats(df, args.label, "Filtered")

print('Loading feature list and formating data')
feature_names = load_feat(df, args.label, 'test', args.feats, model_name)

print('loading encoder')
label_encoder = load_encoder(args.label)

X, Y, _, _ = format_data(df, args.label, feature_names, label_encoder, class_weight = 0)

print(f'Classes: {list(set(Y))}')
print(f'Number of features : {len(feature_names)}')

extract_shapley_values(X, Y, feature_names, label_encoder, features = args.ba, mdl_name = model_name, N=args.N)

print('Voila!')