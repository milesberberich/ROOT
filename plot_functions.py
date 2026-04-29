import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def plot_cross_model_results(y_true, y_pred, feature_importances, train_mode, val_mode, class_names=None):
    """
    Creates a dashboard showing Overall Accuracy, Metrics (P, R, F1),
    the Confusion Matrix, and Feature Importances.
    """
    overall_acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    # 1. Format Metrics Table (Precision, Recall, F1-Score)
    # iloc[:3, :3] selects rows (precision, recall, f1) and columns (classes 1, 2, 3)
    report_df = pd.DataFrame(report_dict)[['1', '2', '3']].T
    if class_names:
        report_df.index = class_names
    report_df = report_df.round(2).reset_index().rename(columns={'index': 'Class'})

    # 2. Setup Figure Layout (3 Columns)
    fig, (ax_table, ax_cm, ax_plot) = plt.subplots(1, 3, figsize=(22, 8),
                                                   gridspec_kw={'width_ratios': [0.8, 1, 1.2]})

    # --- Left Side: Table & Accuracy ---
    ax_table.axis('off')
    ax_table.text(0.5, 0.98, f"Overall Accuracy: {overall_acc:.2%}",
                  fontsize=20, fontweight='bold', ha='center', color='darkgreen')

    data_info = f"CV: {train_mode}" if train_mode == val_mode else f"Train: {train_mode} | Val: {val_mode}"
    ax_table.text(0.5, 0.92, data_info, fontsize=12, fontstyle='italic', ha='center')

    the_table = ax_table.table(cellText=report_df.values, colLabels=report_df.columns,
                               loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(11)
    the_table.scale(1.1, 3)

    # --- Middle Side: Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax_cm, values_format='d', colorbar=False)
    ax_cm.set_title('Confusion Matrix', fontsize=16, fontweight='bold')

    # --- Right Side: Feature Importance ---
    feature_importances.sort_values(ascending=True).plot(kind='barh', color='skyblue', ax=ax_plot)
    ax_plot.set_title('Feature Importance', fontsize=16, fontweight='bold')
    ax_plot.set_xlabel('Relative Importance')
    ax_plot.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_model_results(y_true, y_pred, rf_model, feature_names, train_mode, val_mode):
    """
    Standard results dashboard for single-scene validation.
    """
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    class_names = ['1 (Disturbed)', '2 (Deadwood)', '3 (Undisturbed)']
    return plot_cross_model_results(y_true, y_pred, importances, train_mode, val_mode, class_names)