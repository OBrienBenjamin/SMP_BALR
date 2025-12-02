# SMP_BALR
The Speech Modality Presence in BALR speaker embeddings (SMP_BALR) toolkit is designed to evaluate the presence of speech modalities in BALR speaker embeddings trained on an Automatic Speaker Verification (ASV) task. The image below illustrates the toolkit scheme and functions. 

![toolkit_scheme](https://github.com/user-attachments/assets/5db3f9a5-6944-44bf-97fe-63fd8f0dea10)


# Step 0
Extract BALR vectors from your dataset - **to be made available soon**

# Step 1
Format your BALR vectors and dataset labels. This function will output a .csv to your _data_ folder that includes all labels and BALR dimensions. Please examine the dataset provided in '~/data/' so that your labels are similarly formated. The _split_ flag is an option to split your dataset into 'Train' and 'Test' separate .csv files.

python scripts/format_data.py \
 -label_path '/path/to/your/labels.csv' \
 -data_path '/path/to/your/BALR/' \
 -name 'DatasetName' \
 -split 1

# Step 2 (Optional)
Based on the selected speech modality task, e.g., the string set for the _label_ flag, reduce BALR dimensionality via a frequentist method (Stepwise Linear Discriminat Analysis : 'slda') or a probabilistic method (Bayesian LR : 'bayes'). The feature selection methods (${fs_method}) are defined in the 'lib/ba_functions.py' file. This is an option as redundant features are removed prior to model training. A .csv of the selected binary attributes are exported to the 'features/${label}_${fs_method}.csv' folder.

python scripts/feat_sel.py \
 -method ${fs_method} \
 -label ${label} \
 -data_path '~/data/DatasetName_train_data.csv'

An additional option is to reduce selected features further by exporting features common to the different labels and methods. The _include_ and _exclude_ flags accept lists of strings to include or exclude, respectively, in the title of .csv files in the 'features' directory. The _output_ flag accepts lists and should be written in the format '${label}_common' so as to be read when training models.

python scripts/common_feat.py \
 -include ${label1} ${label2} \
 -exclude ${label3} ${label4} \
 -output ${label1}_common ${label2}_common

# Step 3
Select a model ${model} (Multilayer perceptron : 'mlp'; Decision tree : 'dt') and train it. Different training methods are available and are described below. In addition a SHAP explainer is created based on the model. 

python scripts/ba_train.py \
 -feats ${fs_method} \
 -label ${label} \
 -method ${model} \
 -balance ${balance_method} \
 -speakers ${speaker_method} \
 -data_path '~/data/DatasetName_train_data.csv'

  * Setting the _speakers_ flag to 1 removes all speakers that do not include a minimum of one example of each class (default : 0)
  * Setting the _balance_ flag to 1 oversamples each class up to the size of the largest class, distributing the oversampling proportionally across speakers within each class; setting the flag to 2 exports a class weighting dictionary that is used by the model (default : 0)

# Step 4
Test your model on the test dataset.

python scripts/ba_test.py \
 -feats ${fs_method} \
 -label ${label} \
 -method ${model} \
 -model 'model_${label}_${model}_feats_${fs_method}_balance_${balance_method}_speakers_${speaker_method}'\
 -data_path '~/data/DatasetName_test_data.csv'

# Step 5
Examine the contributions of features to model decisions via SHAP. The _N_ flag is the number of Test samples (random) and _ba_ are the list of features to plot shapley values across classes.

python scripts/shap_feat.py \
 -label ${label} \
 -feats ${fs_method} \
 -method ${mode} \
 -N ${num_samples} \
 -ba ${BAx} ${BAy} ${BAz} \
 -model "model_${label}_${model}_feats_${fs_method}_balance_${balance_method}_speakers_${speaker_method}" \
 -data_path '~/data/DatasetName_test_data.csv'

