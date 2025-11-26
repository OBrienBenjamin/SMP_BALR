import argparse
import sys
import os

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from lib.ba_functions import load_data, filter_data, balance_data, detect_irrelevant_features, load_feat, format_data, train_mlp, train_dt
from lib.export_functions import print_class_stats, export_encoder

parse = argparse.ArgumentParser()
parse.add_argument("-method", type=str)
parse.add_argument("-label", type=str)
parse.add_argument("-data_path", type=str, default=None)
parse.add_argument("-feats", type=str, default=None)
parse.add_argument("-speakers", type=int, default=0)
parse.add_argument("-balance", type=int, default=0)
args = parse.parse_args()

# # # init

# # # ATTENTION! !  Desired labels inserted here # # # 
VALID = ['A', 'H', 'N', 'S'] if args.label == 'EmoClass' else None

# # # # # # # # # # # # # # # # # # # # # # # #  
model_name = f'model_{args.label}_{args.method}_feats_{args.feats}_balance_{args.balance}_speakers_{args.speakers}'

print('Loading data . . .')
df = load_data(data_path = args.data_path, step = 'train')
print_class_stats(df, args.label, "Original")

print('Formatting training data . . .')
df = filter_data(df, args.label, valid_labels = VALID, filt_speakers=args.speakers, filt_samples = args.balance)
print_class_stats(df, args.label, "Filtered")

print('Loading feature list and formating data')
feature_names = load_feat(df, args.label, 'train', args.feats, model_name)
X, Y, label_encoder, class_weights_dict = format_data(df, args.label, feature_names, class_weight = args.balance)

print(f'Selected Training method : {args.method}')
print(f'Feature list : {args.feats}')
print(f'Speakers : {args.speakers}')
print(f'Balanced : {args.balance}')

if args.method == 'mlp':
    train_mlp(X, Y, num_classes = len(label_encoder.classes_), N_INPUTS = len(feature_names), class_weight_dict = class_weights_dict, mdl_name = model_name)
    
if args.method == 'dt':
    train_dt(X, Y, class_weight_dict = class_weights_dict, mdl_name = model_name)


print('Export encoder for future use')
export_encoder(label_encoder, args.label)

print('Training complete')
print('Voila!')