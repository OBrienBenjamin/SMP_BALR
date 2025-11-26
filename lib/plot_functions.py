# # # # PLOT FUNCTIONS
import pandas as pd
import numpy as np
import os

import random

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import shap

def save_plot_predictions(all_labels, all_predictions, label_encoder, mdl_name = 'model'):
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    results_dir = os.path.join(project_root, "results")
    plots_dir = os.path.join(project_root, "plots")
    
    all_labels_decoded = label_encoder.inverse_transform(all_labels)
    all_predictions_decoded = label_encoder.inverse_transform(all_predictions)

    print(classification_report(all_labels_decoded, all_predictions_decoded))
    report = classification_report(all_labels_decoded, all_predictions_decoded, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(results_dir, f"{mdl_name}_classification_report.csv"), index=True)

    conf_matrix = confusion_matrix(all_labels_decoded, all_predictions_decoded)
    row_sums = conf_matrix.sum(axis=1)
    normalized_conf_matrix = conf_matrix / row_sums[:, np.newaxis]

    sorted_labels = label_encoder.classes_
    sorted_conf_matrix = normalized_conf_matrix

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=sorted_conf_matrix,
                                  display_labels=sorted_labels)
    disp.plot()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plots_dir, f"{mdl_name}_confusion_matrix.png"))
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

def plot_shap_dist(feature_names, mdl_name = 'model'):
    project_root = os.path.dirname(os.path.dirname(__file__)) 
    results_dir = os.path.join(project_root, "results")
    plots_dir = os.path.join(project_root, "plots")

    emotions = ["Angry", "Happy", "Neutral", "Sad"]
    dataframes = {}
    for id, emo in enumerate(emotions):
        path = os.path.join(results_dir, f"{mdl_name}_{id}_shap_summary.csv")
        df = pd.read_csv(path, nrows=10) # # # top 10
        dataframes[emo] = df 
            
    emotions = [""] + emotions + [""]

    ba_columns = feature_names

    base_colors = []
    while len(base_colors) < len(ba_columns):
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if color not in base_colors:  # Ensure uniqueness
            base_colors.append(color)
    random.shuffle(base_colors)
    ba_colors = {col: base_colors[i] for i, col in enumerate(ba_columns)}

    y_values = {'Angry': 1, 'Happy': 2, 'Neutral': 3, 'Sad': 4}

    fig, ax = plt.subplots(figsize=(12, 6))
    for emotion, df in dataframes.items():
        numeric_cols = ['Positive_Mean', 'Negative_Mean', 'Max_Mean']
    
    for emo, df in dataframes.items():
        plot_data = []
        for idx, row in df.iterrows():
            ba_name = row['BA']
            for col in numeric_cols:
                value = row[col]
                if pd.notna(value):
                    value = float(value)
                    rounded_value = round(value, 5)
                    plot_data.append({
                        'BA': ba_name,
                        'Value': rounded_value,
                        'Emotion': emo,
                        'Type': col
                    })

        if plot_data:
            grouped_df = pd.DataFrame(plot_data)
            grouped_df['Y'] = y_values[emo] + np.random.uniform(-0.1, 0.1, size=len(grouped_df))
            grouped_df['Color'] = grouped_df['BA'].map(ba_colors)

            for ba_name in grouped_df['BA'].unique():
                subset = grouped_df[grouped_df['BA'] == ba_name]
                ax.scatter(subset['Value'], subset['Y'], s=80, c=subset['Color'], alpha=0.7, label=ba_name)

    # ax.set_xlim(-0.25, 0.25)
    ax.set_ylabel('Emotion')
    ax.set_xlabel('Projected Shapley values')
    ax.set_ylim(0, len(emotions) - 1)

    ax.set_yticks([y_values[emotion] for emotion in emotions if emotion != '']) 
    ax.set_yticklabels([emotion for emotion in emotions if emotion != '']) 
    
    dummy_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=label) for color, label in zip(ba_colors.values(), ba_columns)]
    ax.legend(handles=dummy_handles, title="BA", loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{os.path.join(plots_dir, f"{mdl_name}_shap_top_10.png")}')
    plt.close()