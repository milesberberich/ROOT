import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

#############################################
####### plot_cross_model_results() ##########
#############################################

def plot_cross_model_results(y_true, y_pred, feature_importances, train_mode, val_mode, class_names=None):
    """
    Creates a combined dashboard showing Overall Accuracy,
    a Classification Report table, and Feature Importances.
    """
    from sklearn.metrics import classification_report, accuracy_score

    overall_acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    # 2. Format the Classification Table
    # We slice to get the first 3 classes and exclude 'accuracy', 'macro avg', etc.
    report_df = pd.DataFrame(report_dict).iloc[:-3, :3].T

    if class_names:
        report_df.index = class_names

    report_df = report_df.round(2).reset_index().rename(columns={'index': 'Class'})

    # 3. Dynamic Info Line Logic
    if train_mode == val_mode:
        data_info = f"Trained & Validated on: {train_mode} (CV Split)"
    else:
        data_info = f"Trained on: {train_mode} | Validated with: {val_mode}"

    # 4. Create Combined Figure
    fig, (ax_table, ax_plot) = plt.subplots(1, 2, figsize=(16, 8),
                                            gridspec_kw={'width_ratios': [1, 1.2]})

    # --- Left Side: Table & Accuracy Score ---
    ax_table.axis('off')

    # Big Overall Accuracy Score
    ax_table.text(0.5, 0.98, f"Overall Accuracy: {overall_acc:.2%}",
                  fontsize=22, fontweight='bold', ha='center', color='darkgreen')

    # Dynamic Data Source Line
    ax_table.text(0.5, 0.90, data_info,
                  fontsize=14, fontstyle='italic', ha='center', color='black')

    # Create the Table
    the_table = ax_table.table(cellText=report_df.values,
                               colLabels=report_df.columns,
                               loc='center',
                               cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.1, 3)

    # --- Right Side: Feature Importance ---
    feature_importances.sort_values(ascending=True).plot(kind='barh', color='skyblue', ax=ax_plot)
    ax_plot.set_title('Feature Importance', fontsize=16, fontweight='bold')
    ax_plot.set_xlabel('Relative Importance')
    ax_plot.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


#############################################
########## plot_model_results() #############
#############################################
def plot_model_results(y_true, y_pred, rf_model, feature_names, train_mode, val_mode):
    """
    Generates a performance dashboard: Accuracy, Metrics Table, and Feature Importance.
    """
    # 1. Calculate Metrics
    overall_acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    # 2. Create the Metrics Table (Specific to your 3 classes)
    # We take the first 3 rows and 3 columns (precision, recall, f1-score)
    report_df = pd.DataFrame(report_dict).iloc[:3, :3].T
    report_df.index = ['1 (Disturbed)', '2 (Deadwood)', '3 (Undisturbed)']
    report_df = report_df.round(2).reset_index().rename(columns={'index': 'Class'})

    # 3. Prepare Feature Importance
    importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)

    # 4. Dynamic Info Line
    if train_mode == val_mode:
        data_info = f"Trained & Validated on: {train_mode} (CV Split)"
    else:
        data_info = f"Trained on: {train_mode} | Validated with: {val_mode}"

    # 5. Create Combined Figure
    fig, (ax_table, ax_plot) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1.2]})

    # --- Left Side: Accuracy & Table ---
    ax_table.axis('off')

    # Large Accuracy Display
    ax_table.text(0.5, 0.98, f"Overall Accuracy: {overall_acc:.2%}",
                  fontsize=22, fontweight='bold', ha='center', color='darkgreen')

    # Data Info Line
    ax_table.text(0.5, 0.90, data_info,
                  fontsize=14, fontstyle='italic', ha='center', color='black')

    # Metrics Table
    the_table = ax_table.table(cellText=report_df.values,
                               colLabels=report_df.columns,
                               loc='center',
                               cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.1, 3)

    # --- Right Side: Feature Importance ---
    importances.plot(kind='barh', color='skyblue', ax=ax_plot)
    ax_plot.set_title('Feature Importance', fontsize=16, fontweight='bold')
    ax_plot.set_xlabel('Relative Importance')
    ax_plot.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig