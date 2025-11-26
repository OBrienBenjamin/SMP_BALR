import pandas as pd
import numpy as np

from pathlib import Path

import sys
import os

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from lib.export_functions import print_global, export_shap_values, distribution_before_balance
from lib.model_definitions import DeeperNeuNet, ModelWrapper
from lib.plot_functions import save_plot_predictions, plot_shap_values

import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

from scipy.linalg import det
from scipy.stats import f
from concurrent.futures import ProcessPoolExecutor, as_completed

import shap

import pymc as pm

# # # # # # # # # #  # # # # # 

# # # extract, format balr vectors
def extract_balr(path):
    directory = Path(path)

    rows = []
    for file_path in directory.glob("*_ba.txt"):
        df = pd.read_csv(file_path, header=None, sep=r"\s+")
        
        flipped = [1 if x == 0 else 0 for x in df[0].astype(float)]
        file_name = file_path.name.replace('_ba.txt', '.wav')
        
        rows.append([file_name] + flipped)
    
    ba_df = pd.DataFrame(rows)
    ba_df.rename(columns={0: 'FileName'}, inplace = True)
    ba_df.columns = ['FileName'] + [f'BA{i}' for i in range(ba_df.shape[1] - 1)]
    
    return ba_df

# # # data processing functions
def load_data(data_path=None, step = None):
    if data_path == None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        if step == 'train':
            path = os.path.join(project_root, "data", "msp_train_wespeaker_ba.csv")        
        if step == 'test':
            path = os.path.join(project_root, "data", "msp_development_wespeaker_ba.csv")  
            
        df = pd.read_csv(path)
        print(path)
    else:
        df = pd.read_csv(data_path)
        print(data_path)
        print('NOTE : please make sure your data is formatted correctly (see example in data folder)')
    return df

def filter_data(df, label, valid_labels = None, filt_speakers = 0, filt_samples=0):
    filt_df = df.copy()
        
    if valid_labels != None:
        filt_df = filt_df[filt_df[label].isin(valid_labels)].copy()
    
    if filt_speakers:
        speaker_emotions = df.groupby("Speaker")[label].nunique()
        valid_speakers = speaker_emotions[speaker_emotions == filt_df[label].nunique()].index
        
        print(len(valid_speakers))
        filt_df = filt_df[filt_df["Speaker"].isin(valid_speakers)].copy()
        
    if filt_samples == 1:
        print('oversampling')
        filt_df = balance_data(filt_df, label)
    else:
        filt_df = filt_df
            
    distribution_before_balance(df, filt_df)
    
    return filt_df

def balance_data(df, label):
    class_sizes = df[label].value_counts()
    target_size = class_sizes.max()

    indices = []
    for cls in class_sizes.index:
        cls_df = df[df[label] == cls]

        cur_n = len(cls_df)
        add_n = target_size - cur_n

        if add_n <= 0:
            # already the largest class
            indices.extend(cls_df.index)
            continue

        spk_counts = cls_df["Speaker"].value_counts()
        spk_probs = spk_counts / spk_counts.sum()

        indices.extend(cls_df.index)

        for spk, prob in spk_probs.items():
            spk_subset = cls_df[cls_df["Speaker"] == spk]
            spk_add = int(prob * add_n)
            
            if spk_add == 0:
                continue

            new_samples = resample(
                spk_subset,
                replace=True,
                n_samples=spk_add,
                random_state=42
            )

            indices.extend(new_samples.index)
            
    return df.loc[indices].reset_index(drop=True)

def detect_irrelevant_features(df, mdl_name = 'model'):
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    feats_dir = os.path.join(project_root, "features")
    
    X_df = df[[col for col in df.columns if 'BA' in col]]

    feature_names = []
    remove_list = []
    for col in X_df.columns:
        if X_df[col].nunique() > 1:
            feature_names.append(col)
        else:
            remove_list.append(col)

    print(f'Number of features kept : {len(feature_names)} \tremoved : {len(remove_list)}')
    
    df_feat = pd.DataFrame({"Feature": feature_names})
    df_feat.to_csv(os.path.join(feats_dir, f'{mdl_name}_features.csv'), index = False)
    
    return df_feat

def load_feat(df, label, step = 'train', fs_method = None, mdl_name = 'model'):
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    feats_dir = os.path.join(project_root, "features")
    
    print(fs_method)
    
    if fs_method == None:
        if step == 'train':
            print('Removing redundant features')
            df_feat = detect_irrelevant_features(df, mdl_name)
        else: 
            df_feat = pd.read_csv(os.path.join(feats_dir, f'{mdl_name}_features.csv'))
            print(f'Loading irrelevant features : {len(df_feat)}')
    else:
        df_feat = pd.read_csv(os.path.join(feats_dir, f'{label}_{fs_method}_features.csv'))
        print(f'Loading features identified with {fs_method} method : {len(df_feat)}')
        
    return df_feat["Feature"].tolist()
    
def format_data(df, label, feature_names, label_encoder = None, class_weight = 0):
    class_weights_dict = {}
    
    X = df[feature_names]
    Y = df[label]
    
    if label_encoder == None:
        label_encoder = LabelEncoder()
        LABEL = sorted(list(set(df[label])))
        label_encoder.fit(LABEL)

    Y = label_encoder.transform(Y)

    X = np.array(X.values)
    Y = np.array(Y)
    
    if class_weight == 2:
        print('Getting class weights')
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}    
    
    return X, Y, label_encoder, class_weights_dict

# # # # # # # # # #  # # # # # 

# # # model training functions
def train_mlp(X, Y, N_INPUTS = 512, mdl_name = 'model', num_classes=4, class_weight_dict = {}, n_epochs = 100, batch_size = 32, learning_rate=0.001):
    if class_weight_dict:
        weights = [class_weight_dict[i] for i in sorted(class_weight_dict.keys())]
        class_weights_tensor = torch.tensor(weights, dtype=torch.float).cuda()
    
    # # # model output
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    models_dir = os.path.join(project_root, "models")
    
    mdl_filename = mdl_name + '.pth'
    
    # # # conver to tensors
    X_train = torch.tensor(X, dtype=torch.float32)
    Y_train = torch.tensor(Y, dtype=torch.long)

    train_dataset = TensorDataset(X_train, Y_train)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # # # split train/validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create the model, loss function, and optimizer
    model = DeeperNeuNet(N_INPUTS, num_classes).cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    if class_weight_dict:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).cuda()
    else: 
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    print('Training MLP model . . . ')
    best_val_loss = float('inf')
    train_accuracy = []
    val_accuracy = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for data, label in trainloader:
            data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total_predictions += label.size(0)
            correct_predictions += (predicted == label).sum().item()
        
        train_loss = total_loss / len(trainloader.dataset)
        accuracy = 100 * correct_predictions / total_predictions
        train_accuracy.append(accuracy)

        # Validation step
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        validation_loss = 0.0

        with torch.no_grad():
            for data, label in validationloader:
                data, label = data.float().cuda(), label.cuda()            
                output = model(data)
                
                loss = criterion(output, label)
                validation_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                total_predictions += label.size(0)
                correct_predictions += (predicted == label).sum().item()
                        
        validation_loss /= len(validationloader.dataset)
        validation_accuracy = 100 * correct_predictions / total_predictions
        val_accuracy.append(validation_accuracy)

        print(f'Epoch [{epoch + 1}/{n_epochs}], '
            f'Training Loss: {train_loss:.4f}, '
            f'Training Accuracy: {accuracy:.2f}%, '
            f'Validation Loss: {validation_loss:.4f}, '
            f'Validation Accuracy: {validation_accuracy:.2f}%')

        # Save the best model
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_model_state = model.state_dict()
            
            save_path = os.path.join(models_dir, mdl_filename)
            torch.save(best_model_state, save_path)
            
            print("Best model saved.")

        scheduler.step()
        
    best_model = DeeperNeuNet(N_INPUTS, num_classes).cuda()
    best_model.load_state_dict(best_model_state)
    best_model.eval()

    model_wrapper = ModelWrapper(best_model)
    predict_fn = model_wrapper
    
    background = np.array(X)
    background = np.unique(background, axis=0)
    n_background = min(200, background.shape[0])
    idx = np.random.choice(background.shape[0], n_background, replace=False)
    background = background[idx]
    
    explainer = shap.KernelExplainer(predict_fn, background)
    with open(os.path.join(models_dir, f'{mdl_name}_explainer.pkl'), 'wb') as file:
        pickle.dump(explainer, file)

def train_dt(X, Y, class_weight_dict = {}, mdl_name = 'model'):
    # # # model output
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    models_dir = os.path.join(project_root, "models")
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, Y,
        test_size=0.2,
        stratify=Y,
        shuffle=True,
        random_state=42
    )
    
    param_grid = {
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"],
    }
    
    mdl_filename = mdl_name + '.pth'        
    FOLDS = 5
    
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42, class_weight=class_weight_dict),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    mdl = grid.best_estimator_

    print("Best params:", grid.best_params_)
    print("Train CV score:", grid.best_score_)
    
    save_path = os.path.join(models_dir, mdl_filename)        
    with open(save_path,'wb') as f:
        pickle.dump(mdl,f)
        
    test_pred = mdl.predict(X_valid)
    test_acc = accuracy_score(y_valid, test_pred)
    print("Validation accuracy:", test_acc)
    
    # # #  explainer
    explainer = shap.TreeExplainer(mdl)
    explainer_path = os.path.join(models_dir, f"{mdl_name}_explainer.pkl")
    with open(explainer_path, 'wb') as f:
        pickle.dump(explainer, f)

# # # # # # # # # #  # # # # # 

# # # model testing functions    
def test_mlp(X, Y, feature_names, label_encoder, mdl_name = 'model', explain = 0, num_classes=4, n_epochs = 100, batch_size = 32, learning_rate=0.001):
    # # # I/O
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    
    mdl_filename = mdl_name + '.pth'
    
    X_test = torch.tensor(X, dtype=torch.float32)
    Y_test = torch.tensor(Y, dtype=torch.long)

    test_dataset = TensorDataset(X_test, Y_test)

    print('loading model . . .')
    model = DeeperNeuNet(len(feature_names), num_classes).cuda()
    model_path = os.path.join(models_dir, mdl_filename)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_correct_predictions = 0
    test_total_predictions = 0
    all_labels = []
    all_predictions = []
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.float().cuda(), label.cuda()
            output = model(data)
            _, predicted = torch.max(output, 1)
            test_total_predictions += label.size(0)
            test_correct_predictions += (predicted == label).sum().item()
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_accuracy = 100 * test_correct_predictions / test_total_predictions
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    results_path = os.path.join(results_dir, f"{mdl_name}_results.csv")

    # # export
    df_labels_predictions = pd.DataFrame({
        'True_Label': all_labels,
        'Predicted_Label': all_predictions
    })
    df_labels_predictions.to_csv(results_path, index=False)

    # # # export classification report, plots
    save_plot_predictions(all_labels, all_predictions, label_encoder, mdl_name)

    # # # if explaining model
    if explain:
        print('Explaining model. . .')
        explainer_path = os.path.join(models_dir, f"{mdl_name}_explainer.pkl")

        with open(explainer_path, "rb") as file:
            explainer = pickle.load(file)
            
        X_test_np = np.array(X_test)
        Y_test_np = np.array(Y_test)
        
        idx = np.random.choice(len(X_test_np), size=50, replace=False)

        X_sample = X_test_np[idx]
        Y_sample = Y_test_np[idx]
        
        shap_values = explainer.shap_values(X_sample) # # # non-batch (works!)
        # shap_values = batch_shap_values(explainer, X_test_np, batch_size=50) # haven't tried

        print('Global importance')
        print_global(shap_values, feature_names)
        
        export_shap_values(shap_values, X_sample, Y_sample, label_encoder, feature_names, mdl_name)

        # plot_shap_dist(mdl_name) # # # does not function (maybe make it into another function)

        # # # create typical SHAP plot for each emotion
        plot_shap_values(shap_values, X_sample, Y_sample, label_encoder, feature_names, mdl_name)
        
def test_dt(X, Y, feature_names, label_encoder, mdl_name = 'model', explain = 0):
    # # # I/O
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
        
    model_path = os.path.join(models_dir, mdl_name + ".pth")
    print(f"Loading Decision Tree model from: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict and compute accuracy
    y_pred = model.predict(X)
    test_acc = accuracy_score(Y, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")

    results_path = os.path.join(results_dir, f"{mdl_name}_results.csv")

    # # export true, predicted labels    
    df_labels_predictions = pd.DataFrame({
        'True_Label': Y,
        'Predicted_Label': y_pred
    })
    df_labels_predictions.to_csv(results_path, index=False)

    # # # export classification report, plots
    save_plot_predictions(Y, y_pred, label_encoder, mdl_name)

    # SHAP EXPLAINER
    if explain:
        print("Loading SHAP explainer…")
        explainer_path = os.path.join(models_dir, f"{mdl_name}_explainer.pkl")

        with open(explainer_path, "rb") as f:
            explainer = pickle.load(f)

        print("Computing SHAP values…")
        shap_values = explainer.shap_values(X)

        print('Global importance')
        print_global(shap_values, feature_names)
        
        export_shap_values(shap_values, X, Y, label_encoder, feature_names, mdl_name)

        # plot_shap_dist(mdl_name) # # # does not function (maybe make it into another function)

        # # # create typical SHAP plot for each emotion
        plot_shap_values(shap_values, X, Y, label_encoder, feature_names, mdl_name)

# # # # # # # # # #  # # # # # 

# # # feature selection functions
def sel_feat_bayes(X, Y, feature_names, label, K=None, top_features=50, n_draws=1000):
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    results_dir = os.path.join(project_root, "results")
    
    _, n_features = X.shape
    if K is None:
        K = len(np.unique(Y))
    if top_features is None:
        top_features = n_features

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=0.5, shape=(n_features, K))
        intercept = pm.Normal("intercept", mu=0, sigma=1, shape=K)

        logits = pm.math.dot(X, beta) + intercept
        pm.Categorical("y_obs", logit_p=logits, observed=Y)

        approx = pm.fit(n=5000, method="advi")
        trace = approx.sample(draws=n_draws)

    beta_post = trace.posterior["beta"].stack(draws=("chain", "draw")).values
    if beta_post.ndim == 2:
        beta_post = beta_post[:, np.newaxis, :]  # handle single-class case

    prob_pos = (beta_post > 0).mean(axis=2)  # shape: (features, classes)
    prob_any_class = 1 - np.prod(1 - prob_pos, axis=1)

    ranked_idx = np.argsort(prob_any_class)[::-1][:top_features]

    n_classes = beta_post.shape[1]
    mean_beta = beta_post.mean(axis=2)
    df_features = pd.DataFrame({
        "Feature": np.repeat(feature_names, n_classes),
        "Class": np.tile(np.arange(n_classes), n_features),
        "Probability": prob_pos.flatten(),
        "MeanBeta": mean_beta.flatten()
    })
    df_features.to_csv(os.path.join(results_dir, f'{label}_bayes_feature_probs.csv'), index=False)
    
    top_feature_names = [feature_names[i] for i in ranked_idx]
    top_probs = prob_any_class[ranked_idx]
    df_top = pd.DataFrame({
        "Feature": top_feature_names,
        "Probability": top_probs
    })
    
    return df_top

def sel_feat_slda(X, y, feature_names, p_thresh=0.05, workers=8, batch_size=16):
    selected = []
    remaining = list(range(X.shape[1]))
    history = []

    # Initialize parallel environment
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initializer,
        initargs=(X, y)
    ) as executor:

        while True:
            futures = {}

            # Submit tasks in batches
            for batch in chunked(remaining, batch_size):
                for idx in batch:
                    futures[executor.submit(evaluate_feature, idx, selected)] = idx

                # Collect finished results batch-wise
                best = (1.0, None, 1.0)  # lam, idx, p

                for fut in as_completed(list(futures.keys())):
                    idx = futures[fut]
                    try:
                        idx2, (lam, F, p_val) = fut.result()
                    except:
                        continue

                    if p_val < p_thresh and lam < best[0]:
                        best = (lam, idx2, p_val)

                # If we found a feature in this batch, stop early
                if best[1] is not None:
                    break

            lam, feat_idx, p_val = best

            # No more improvements
            if feat_idx is None:
                break

            # Store selected feature
            selected.append(feat_idx)
            remaining.remove(feat_idx)

            history.append({
                "Feature": feature_names[feat_idx],
                "Wilks_Lambda": lam,
                "P_Value": p_val
            })

            print(f"Selected: {feature_names[feat_idx]}  (λ={lam:.4f}, p={p_val:.4g})")

    if not history:
        return pd.DataFrame(columns=["Feature", "Wilks_Lambda", "P_Value"])

    df = pd.DataFrame(history)
    
    return df.sort_values("Wilks_Lambda", ascending=False).reset_index(drop=True)

def evaluate_feature(idx, selected):
    cols = selected + [idx]
    X_sub = X_global[:, cols]
    
    return idx, compute_wilks_lambda(X_sub, y_global)

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def initializer(X_vals_, y_vals_):
    global X_global, y_global
    X_global = X_vals_
    y_global = y_vals_

def compute_wilks_lambda(X, y):
    classes = np.unique(y)
    n_total = len(y)
    n_features = X.shape[1]
    k = len(classes)

    overall_mean = np.mean(X, axis=0)

    # Within-class scatter
    Sw = np.zeros((n_features, n_features))
    for cls in classes:
        X_cls = X[y == cls]
        m = np.mean(X_cls, axis=0)
        Xc = X_cls - m
        Sw += np.einsum("ij,ik->jk", Xc, Xc)

    # Total scatter
    XcT = X - overall_mean
    St = np.einsum("ij,ik->jk", XcT, XcT)

    det_St = det(St)
    det_Sw = det(Sw)
    lam = det_Sw / det_St if det_St != 0 else 1.0

    df1 = n_features * (k - 1)
    df2 = n_total - 1 - (n_features + k)/2

    if lam == 0 or lam == 1 or df2 <= 0:
        return lam, np.nan, 1.0

    F_stat = ((1 - lam) / lam) * (df2 / df1)
    p_val = f.sf(F_stat, df1, df2)
    return lam, F_stat, p_val