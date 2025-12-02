import pandas as pd
import numpy as np
import os

import pickle

def export_encoder(label_encoder, label):
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    models_dir = os.path.join(project_root, "models")
    
    with open(os.path.join(models_dir, f"{label}_label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    
def load_encoder(label):
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    models_dir = os.path.join(project_root, "models")
    
    with open(os.path.join(models_dir, f"{label}_label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
        
    return label_encoder
    
def distribution_before_balance(original, filter):
    original_size = len(original)
    remaining_size = len(filter)
    percentage_retained = (remaining_size / original_size) * 100

    original_speakers = original["Speaker"].nunique()
    remaining_speakers = filter["Speaker"].nunique()
    percentage_speakers_retained = (remaining_speakers / original_speakers) * 100

    print(f"Original rows: {original_size}")
    print(f"Remaining rows: {remaining_size}")
    print(f"Percentage of rows retained: {percentage_retained:.2f}%\n")

    print(f"Original speakers: {original_speakers}")
    print(f"Remaining speakers: {remaining_speakers}")
    print(f"Percentage of speakers retained: {percentage_speakers_retained:.2f}%")
    
    return

def print_class_stats(df, label = 'EmoClass', name="Dataset"):
    print(f"\n--- {name} Stats ---")
    classes = df[label].unique()
    for cls in sorted(classes):
        cls_df = df[df[label] == cls]
        n_samples = len(cls_df)
        n_speakers = cls_df['Speaker'].nunique()
        avg_per_speaker = n_samples / n_speakers
        
        print(f"{label} {cls}: {n_samples} samples, {n_speakers} Speakers, avg {avg_per_speaker:.1f} per speaker")
        
    return

def export_shap_values(shap_values, X, Y, label_encoder, feature_names, mdl_name = 'model'):
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    results_dir = os.path.join(project_root, "results")

    # Decode Y into the original class names
    Y_decoded = label_encoder.inverse_transform(Y)

    present_classes = np.unique(Y_decoded)
    class_mask = np.isin(label_encoder.classes_, present_classes)

    filtered_shap_values = [sv for sv, keep in zip(shap_values, class_mask) if keep]
    filtered_classes = label_encoder.classes_[class_mask]

    print("Encoded Y unique:", np.unique(Y))
    print("Label encoder classes:", label_encoder.classes_)
    print("Mask (correct):", class_mask)

    for class_label, class_shap in zip(filtered_classes, filtered_shap_values):
        df = pd.DataFrame(class_shap, columns=feature_names)
        df.insert(0, 'True_Label', Y)
        df.insert(0, 'Sample_ID', np.arange(len(X)))

        csv_path = os.path.join(results_dir, f"{mdl_name}_shap_values_class_{class_label}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved raw SHAP values for class {class_label} → {csv_path}")
        
        pos_mean = np.nanmean(np.where(class_shap > 0, class_shap, np.nan), axis=0)
        neg_mean = np.nanmean(np.where(class_shap < 0, class_shap, np.nan), axis=0)
        
        combined = np.vstack([
            np.nan_to_num(pos_mean, nan=0.0),  # replace NaN with 0
            np.nan_to_num(neg_mean, nan=0.0)
        ])

        indices = np.argmax(np.abs(combined), axis=0)
        max_abs_with_sign = combined[indices, np.arange(combined.shape[1])]

        all_nan_mask = np.isnan(pos_mean) & np.isnan(neg_mean)
        max_abs_with_sign[all_nan_mask] = np.nan

        ranking_df = pd.DataFrame({
            "BA": feature_names,
            "Positive_Mean": pos_mean,
            "Negative_Mean": neg_mean,
            "Max_Mean": max_abs_with_sign
        })
        ranking_df.sort_values(by="Max_Mean", ascending=False, inplace=True)

        # export
        csv_path = os.path.join(results_dir, f"{mdl_name}_{class_label}_shap_summary.csv")
        ranking_df.to_csv(csv_path, index=False)

def load_shapley_values(mdl_name, label):
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    results_dir = os.path.join(project_root, "results")
    
    csv_path = os.path.join(results_dir, f"{mdl_name}_shap_values_class_{label}.csv")
    df_shap = pd.read_csv(csv_path)    
    return df_shap

def print_global(shap_values, feature_names):
    global_importance = np.mean(np.abs(shap_values), axis=(0, 1))

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": global_importance
    })

    importance_df.sort_values("Importance", ascending=False, inplace=True)
    importance_df.reset_index(drop=True, inplace=True)
    total_importance = importance_df["Importance"].sum()

    top5 = importance_df.head(5).copy()
    top5_pct = 100 * top5["Importance"].sum() / total_importance

    print("\n=== TOP 5 FEATURES (GLOBAL SHAP) ===")
    for i, row in top5.iterrows():
        pct = 100 * row["Importance"] / total_importance
        print(f"{row['Feature']}: {row['Importance']:.6f}  ({pct:.2f}%)")
    print(f"Total Top-5 Contribution: {top5_pct:.2f}%")

    top10 = importance_df.head(10).copy()
    top10_pct = 100 * top10["Importance"].sum() / total_importance

    print("\n=== TOP 10 FEATURES (GLOBAL SHAP) ===")
    for i, row in top10.iterrows():
        pct = 100 * row["Importance"] / total_importance
        print(f"{row['Feature']}: {row['Importance']:.6f}  ({pct:.2f}%)")
    print(f"Total Top-10 Contribution: {top10_pct:.2f}%")
    
def export_fs_results(df, label, fs_method):
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    feats_dir = os.path.join(project_root, "features")

    df.to_csv(
        os.path.join(feats_dir, f'{label}_{fs_method}_features.csv'),
        index=False
    )
