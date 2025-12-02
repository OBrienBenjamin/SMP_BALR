# # # # PLOT FUNCTIONS
import pandas as pd
import numpy as np
import os

import random

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import shap

def save_plot_predictions(all_labels, all_predictions, label_encoder, mdl_name='model'):
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    results_dir = os.path.join(project_root, "results")
    plots_dir = os.path.join(project_root, "plots")

    y_true = label_encoder.inverse_transform(all_labels)
    y_pred = label_encoder.inverse_transform(all_predictions)

    true_classes = np.unique(y_true)

    INVALID = "__INVALID__"
    y_pred_filtered = np.where(np.isin(y_pred, true_classes), y_pred, INVALID)

    print(classification_report(
        y_true,
        y_pred_filtered,
        labels=true_classes,     # excludes INVALID completely
        zero_division=0
    ))

    report = classification_report(
        y_true,
        y_pred_filtered,
        labels=true_classes,
        output_dict=True,
        zero_division=0
    )
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(results_dir, f"{mdl_name}_classification_report.csv"))

    df = pd.DataFrame({"true": y_true, "pred": y_pred_filtered})
    cm = pd.crosstab(df["true"], df["pred"])

    if INVALID in cm.columns:
        cm.drop(columns=[INVALID], inplace=True)

    cm = cm.reindex(index=true_classes, columns=true_classes, fill_value=0)
    cm_norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm.to_numpy(),
        display_labels=true_classes
    )
    disp.plot(values_format=".2f", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(os.path.join(plots_dir, f"{mdl_name}_confusion_matrix.png"),
                bbox_inches="tight", dpi=300)
    plt.close()


def plot_shap_values(shap_values, X, Y, label_encoder, feature_names, mdl_name):
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    plots_dir = os.path.join(project_root, "plots")

    Y_decoded = label_encoder.inverse_transform(Y)

    present_classes = np.unique(Y_decoded)
    class_mask = np.isin(label_encoder.classes_, present_classes)

    filtered_shap_values = [sv for sv, keep in zip(shap_values, class_mask) if keep]
    filtered_classes = label_encoder.classes_[class_mask]

    print("Decoded Y unique:", present_classes)
    print("Label encoder classes:", label_encoder.classes_)
    print("Mask:", class_mask)
    print("Filtered classes:", filtered_classes)

    for class_label, class_shap in zip(filtered_classes, filtered_shap_values):

        print(f"SHAP summary plot for {class_label}")

        shap.summary_plot(
            class_shap,
            X,
            feature_names=feature_names,
            show=False,
            plot_size=(10, 8)
        )

        plt.title(f"SHAP Summary Plot - {class_label}")
        plt.tight_layout()

        save_path = os.path.join(plots_dir, f"{mdl_name}_{class_label}_shap.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Saved: {save_path}")
        
def plot_shap_feature_across_classes(shap_values,  X, Y, label_encoder, feature_names,feature, mdl_name="model"):
    project_root = os.path.dirname(os.path.dirname(__file__))
    plots_dir = os.path.join(project_root, "plots")

    # Decode Y -> class names present
    Y_decoded = label_encoder.inverse_transform(Y)
    present_classes = np.unique(Y_decoded)

    # Determine feature index
    if isinstance(feature, str):
        if feature not in feature_names:
            raise ValueError(f"Feature '{feature}' not found.")
        f_idx = feature_names.index(feature)
        f_name = feature
    else:
        f_idx = feature
        f_name = feature_names[f_idx]

    # Feature values for coloring
    feat_vals = X[:, f_idx]

    # Normalize feature values (like SHAP)
    norm = Normalize(vmin=np.nanpercentile(feat_vals, 5),
                     vmax=np.nanpercentile(feat_vals, 95))
    cmap = get_cmap("RdBu_r")  # SHAP uses reversed RdBu

    # Filter shap values to present classes
    class_mask = np.isin(label_encoder.classes_, present_classes)
    filtered_shap_values = [sv for sv, keep in zip(shap_values, class_mask) if keep]
    filtered_classes = label_encoder.classes_[class_mask]

    # Extract SHAP values for this feature
    feature_shap_by_class = [sv[:, f_idx] for sv in filtered_shap_values]

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 6))  # create explicit axes

    for i, (cls, shap_vals) in enumerate(zip(filtered_classes, feature_shap_by_class)):
        y_loc = np.full_like(shap_vals, i, dtype=float)
        colors = cmap(norm(feat_vals))
        ax.scatter(
            shap_vals,
            y_loc,
            s=14,
            alpha=0.75,
            c=colors
        )

    # Formatting
    ax.set_yticks(range(len(filtered_classes)))
    ax.set_yticklabels(filtered_classes)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(f"SHAP value for '{f_name}'")
    ax.set_ylabel("Class")
    ax.set_title(f"SHAP Feature Importance Across Classes\nFeature: {f_name}")

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required by matplotlib
    fig.colorbar(sm, ax=ax, label=f"Feature value: {f_name}")

    plt.tight_layout()

    save_path = os.path.join(
        plots_dir,
        f"{mdl_name}_feature_{f_name}_shap_across_classes.png"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

def batch_shap_values(explainer, X, batch_size=50):
    shap_list = None
    
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        print(f"Batch {i}–{i+len(batch)-1}")

        batch_shap = explainer.shap_values(batch)

        if shap_list is None:
            shap_list = [b.copy() for b in batch_shap]   # init list for classes
        else:
            for k in range(len(shap_list)):
                shap_list[k] = np.vstack([shap_list[k], batch_shap[k]])

    return shap_list



# # # # # custom function (needs work and use)
# def plot_shap_dist(feature_names, mdl_name = 'model'):
#     project_root = os.path.dirname(os.path.dirname(__file__)) 
#     results_dir = os.path.join(project_root, "results")
#     plots_dir = os.path.join(project_root, "plots")

#     emotions = ["Angry", "Happy", "Neutral", "Sad"]
#     dataframes = {}
#     for id, emo in enumerate(emotions):
#         path = os.path.join(results_dir, f"{mdl_name}_{id}_shap_summary.csv")
#         df = pd.read_csv(path, nrows=10) # # # top 10
#         dataframes[emo] = df 
            
#     emotions = [""] + emotions + [""]

#     ba_columns = feature_names

#     base_colors = []
#     while len(base_colors) < len(ba_columns):
#         color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
#         if color not in base_colors:  # Ensure uniqueness
#             base_colors.append(color)
#     random.shuffle(base_colors)
#     ba_colors = {col: base_colors[i] for i, col in enumerate(ba_columns)}

#     y_values = {'Angry': 1, 'Happy': 2, 'Neutral': 3, 'Sad': 4}

#     fig, ax = plt.subplots(figsize=(12, 6))
#     for emotion, df in dataframes.items():
#         numeric_cols = ['Positive_Mean', 'Negative_Mean', 'Max_Mean']
    
#     for emo, df in dataframes.items():
#         plot_data = []
#         for idx, row in df.iterrows():
#             ba_name = row['BA']
#             for col in numeric_cols:
#                 value = row[col]
#                 if pd.notna(value):
#                     value = float(value)
#                     rounded_value = round(value, 5)
#                     plot_data.append({
#                         'BA': ba_name,
#                         'Value': rounded_value,
#                         'Emotion': emo,
#                         'Type': col
#                     })

#         if plot_data:
#             grouped_df = pd.DataFrame(plot_data)
#             grouped_df['Y'] = y_values[emo] + np.random.uniform(-0.1, 0.1, size=len(grouped_df))
#             grouped_df['Color'] = grouped_df['BA'].map(ba_colors)

#             for ba_name in grouped_df['BA'].unique():
#                 subset = grouped_df[grouped_df['BA'] == ba_name]
#                 ax.scatter(subset['Value'], subset['Y'], s=80, c=subset['Color'], alpha=0.7, label=ba_name)

#     # ax.set_xlim(-0.25, 0.25)
#     ax.set_ylabel('Emotion')
#     ax.set_xlabel('Projected Shapley values')
#     ax.set_ylim(0, len(emotions) - 1)

#     ax.set_yticks([y_values[emotion] for emotion in emotions if emotion != '']) 
#     ax.set_yticklabels([emotion for emotion in emotions if emotion != '']) 
    
#     dummy_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=label) for color, label in zip(ba_colors.values(), ba_columns)]
#     ax.legend(handles=dummy_handles, title="BA", loc='upper left', bbox_to_anchor=(1, 1))
    
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(f'{os.path.join(plots_dir, f"{mdl_name}_shap_top_10.png")}')
#     plt.close()