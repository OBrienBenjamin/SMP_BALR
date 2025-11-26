# SMP_BALR
The Speech Modality Presence in BALR speaker embeddings (SMP_BALR) toolkit is designed to evaluate the presence of speech modalities in BALR speaker embeddings trained on an Automatic Speaker Verification (ASV) task. The image below illustrates the toolkit scheme and functions. 

<img width="973" height="484" alt="toolkit_scheme" src="https://github.com/user-attachments/assets/cabf290f-a8b8-4dc5-b18d-6b240a5e0ca4" />

# Step 0
Extract BALR vectors from your dataset - see [https://gitlab.inria.fr/inria-defense/balr/]

# Step 1
Format your BALR vectors and dataset labels. This function will output a .csv to your _data_ folder that includes all labels and BALR dimensions. Please examine the dataset provided in '~/data/' so that your labels are similarly formated. The split flag is an option to split your dataset into 'Train' and 'Test' separate .csv files.

python scripts/format_data.py -label_path '/path/to/your/labels.csv' -data_path '/path/to/your/BALR/' -name 'DatasetName' -split 1

# Step 2 (Optional)
Based on the selected speech modality task, e.g., {label}, reduce BALR dimensionality via a frequentist method (Stepwise Linear Discriminat Analysis : 'slda') or a probabilistic method (Bayesian LR : 'bayes'). The methods are defined in the 'lib/ba_functions.py' file. This is an option as redudant features are removed prior to model training. A .csv of the selected binary attributes are exported to the 'features/{label}_{method}.csv' folder.

python scripts/feat_sel.py -method 'bayes' -label 'EmoClass' -data_path '~/data/DatasetName_train_data.csv'

# Step 3
Select a model (Multilayer perceptron : 'mlp'; Decision tree : 'dt') and train it. Different training methods are available and are described below. In addition a SHAP explainer is created based on the model. 

python scripts/ba_train.py -feats 'bayes' -label 'EmoClass' -method 'mlp' -balance 2 -speakers 0 -data_path '~/data/DatasetName_train_data.csv'

  * Setting the _speakers_ flag to 1 removes all speakers that do not include a minimum of one example of each class (default : 0)
  * Setting the _balance_ flag to 1 oversamples each class up to the size of the largest class, distributing the oversampling proportionally across speakers within each class; setting the flag to 2 exports a class weighting dictionary that is used by the model (default : 0)

# Step 4
Test your model on the test dataset. Setting the _explain_ flag to 1 exports the top 5 and 10 BALR dimensions that contributed to model decisions (default : 0).

python scripts/ba_test.py -feats 'bayes' -label 'EmoClass' -method 'mlp' -explain 1 -model 'model_Language_mlp_feats_bayes_balance_2_speakers_0' -data_path '~/data/DatasetName_test_data.csv'



