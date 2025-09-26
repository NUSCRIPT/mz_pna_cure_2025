import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, cross_val_score
from sklearn.calibration import calibration_curve

# ==========================================================================================
# Plots
# ==========================================================================================
def prettify_feature_names(columns):
    # handle specific names
    special_tokens = ['gcs', 'fio2', 'po2', 'rass', 'pco2', 'wbc', 'rdw', 'ldh', 'sofa', 'bmi']
    
    pretty_names = []

    for col in columns:
        if col == 'sofa_points_htn':
            pretty_names.append('Cardiovascular SOFA Score')
        elif col.startswith('po2_fio2_ratio_'):
            suffix = col.split('_')[-1].capitalize()
            pretty_names.append(r'PaO$_2$/FiO$_2$ ' + suffix)
        elif col.startswith('po2_art_'):
            suffix = col.split('_')[-1].capitalize()
            pretty_names.append(r'PaO$_2$ ' + suffix)
        elif col.startswith('abs_lymphocytes_'):
            suffix = col.split('_')[-1].capitalize()
            pretty_names.append('Absolute Lymphocytes ' + suffix)
        elif col == 'edw_adm_age':
            pretty_names.append('Age')
        elif col.startswith('pco2_art_'):
            suffix = col.split('_')[-1].capitalize()
            pretty_names.append(r'PaCO$_2$ ' + suffix)
        elif col.startswith('fio2_'):
            suffix = col.split('_')[-1].capitalize()
            pretty_names.append(r'FiO$_2$ ' + suffix)
        else:
            parts = col.split('_')
            pretty_parts = []
            for part in parts:
                if part.lower() in special_tokens:
                    pretty_parts.append(part.upper())
                else:
                    pretty_parts.append(part.capitalize())
            pretty_names.append(' '.join(pretty_parts))
    return pretty_names

# SHAP plot
# Initialize SHAP explainer
def plot_shap(model,data, plot_title, plot=True, save_path=None, max_display=10):
    explainer = shap.Explainer(model)
    shap_values = explainer(data)

    # get cleaner column names
    pretty_names = prettify_feature_names(data.columns)

    # Plot SHAP values
    if plot:
        new_title =  plot_title.replace('_', ' ')
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, data, feature_names=pretty_names, max_display=max_display, show=False)
        plt.title(new_title, fontsize=14)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # plt.show()
        plt.close()
    return shap_values

# ROC plot
def plot_auroc(y_train, y_test, y_train_pred_proba, y_test_pred_proba, title,
               fontsize=14, figsize=(10, 10)):

    y_train_preds = y_train_pred_proba
    y_test_preds = y_test_pred_proba

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # ROC curve data
    fpr1, tpr1, _ = roc_curve(y_train, y_train_preds)
    fpr2, tpr2, _ = roc_curve(y_test, y_test_preds)

    auc1 = roc_auc_score(y_train, y_train_preds)
    auc2 = roc_auc_score(y_test, y_test_preds)


    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.plot(fpr1, tpr1, color='C1', label="Train set (AUROC = "+str(round(auc1, 3))+")")
    ax.plot(fpr2, tpr2, color='C2', label="Test set (AUROC = "+str(round(auc2, 3))+")")

    # Add line for random chance
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random Chance')

    ax.set_ylabel('True Positive Rate', fontsize=fontsize)
    ax.set_xlabel('False Positive Rate', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize - 2)
    ax.tick_params(axis='y', labelsize=fontsize - 2)
    ax.grid(linestyle=':')
    ax.legend(loc='best', fontsize=fontsize - 2)
    ax.set_title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.show()

def plot_shap_customized_axis(model,data, plot_title, 
                              left_label=None, right_label=None,
                              plot=True, save_path=None, 
                              max_display=10):
    explainer = shap.Explainer(model)
    shap_values = explainer(data)

    # get cleaner column names
    pretty_names = prettify_feature_names(data.columns)

    # Plot SHAP values
    if plot:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, data, feature_names=pretty_names,
            max_display=max_display, show=False
        )

        # Axis and colorbar tweaks
        ax = plt.gca()
        fig = plt.gcf()
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=12)
        
        # Colorbar tweak
        if fig.axes and (left_label is not None or right_label is not None):
            colorbar_ax = fig.axes[-1]
            colorbar_ax.tick_params(labelsize=15)
            colorbar_ax.set_yticklabels(['Low/No', 'High/Yes'])
            colorbar_ax.yaxis.label.set_size(10)

            # Custom left/right labels under the colorbar
            if left_label is not None:
                ax.text(-0.1, -0.18, left_label,
                    transform=ax.transAxes,
                    fontsize=15,
                    horizontalalignment='left',
                    fontweight='bold')

            if right_label is not None:
                ax.text(1.1, -0.18, right_label,
                    transform=ax.transAxes,
                    fontsize=15,
                    horizontalalignment='right',
                    fontweight='bold')

        plt.title(plot_title.replace('_', ' '), fontsize=20, pad=20)
        # plt.xlabel('SHAP value (impact on model output)', fontsize=14)
        ax.xaxis.label.set_visible(False)
        plt.tight_layout()
        # plt.subplots_adjust(bottom=0.22) 

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # plt.show()
        plt.close()

# ==========================================================================================
# performance metrics
# ==========================================================================================

def cross_validation_metrics(best_model, X, y, episode_ids, n_splits=5):
    # Set up n-fold cross-validation
    gkf = GroupKFold(n_splits=n_splits)

    # Arrays to store metrics
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    precisions = []
    aps = []
    mean_recall = np.linspace(0, 1, 100)


    for train_index, val_index in gkf.split(X, y, groups=episode_ids):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Fit model
        best_model.fit(X_tr, y_tr)
        
        # Predict probabilities
        y_val_preds = best_model.predict_proba(X_val)[:, 1]
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_val.astype(int), y_val_preds)
        auc_score = auc(fpr, tpr)
        aucs.append(auc_score)
        
        # Interpolate TPR
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        

        # PRC
        precision, recall, _ = precision_recall_curve(y_val.astype(int), y_val_preds)
        ap_score = average_precision_score(y_val.astype(int), y_val_preds)
        aps.append(ap_score)
        # Interpolate precision to mean_recall grid
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])  # flip for increasing recall
        precisions.append(interp_precision)


    # Compute mean TPR and AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0 
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)

    # Compute means/std for PRC
    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)

    return {
        "aurocs": aucs,
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "aps": aps,
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
        "std_precision": std_precision
    }

# get metric results
def get_metrics_dict(results):
    metrics = results['metrics']
    
    accuracy = metrics.get('accuracy', None)
    precision = metrics.get('ppv', None) or metrics.get('precision', None)
    recall = metrics.get('sensitivity', None) or metrics.get('recall', None)
    specificity = metrics.get('specificity', None)
    f1 = metrics.get('f1', None) or metrics.get('f1-score', None)
    auroc = metrics.get('auroc', None)
    aupr = metrics.get('aupr', None)
    npv = metrics.get('npv', None)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-score": f1,
        "AUROC": auroc,
        "AUPR": aupr,
        "NPV": npv,
    }

# ==========================================================================================
# bootstrap
# ==========================================================================================

def bootstrap_metrics(y_true, y_pred_prob, threshold=0.5, n_bootstrap=500, random_state=42):
    """
    Generate bootstrap confidence intervals for classification metrics.
    """
    np.random.seed(random_state)
    
    # Input validation
    if len(y_true) != len(y_pred_prob):
        raise ValueError("y_true and y_pred_prob must have the same length")
    
    n_samples = len(y_true)
    
    # Storage for bootstrap statistics
    bootstrap_aurocs = []
    bootstrap_sensitivities = []
    bootstrap_specificities = []
    bootstrap_ppvs = []
    bootstrap_npvs = []
    bootstrap_f1s = []
    bootstrap_accuracies = []
    
    # Progress tracking
    failed_samples = 0
    
    for i in range(n_bootstrap):
        # Progress indicator (every 100 iterations)
        if (i + 1) % 100 == 0:
            print(f"Bootstrap progress: {i + 1}/{n_bootstrap}")
            
        # Bootstrap sampling with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true.iloc[bootstrap_indices] if hasattr(y_true, 'iloc') else y_true[bootstrap_indices]
        y_pred_prob_boot = y_pred_prob[bootstrap_indices]
        
        # Skip if bootstrap sample has only one class
        if len(np.unique(y_true_boot)) < 2:
            failed_samples += 1
            continue
            
        # Calculate AUROC
        try:
            auroc_boot = roc_auc_score(y_true_boot, y_pred_prob_boot)
            bootstrap_aurocs.append(auroc_boot)
        except (ValueError, IndexError) as e:
            failed_samples += 1
            continue
            
        # Calculate classification metrics at threshold
        y_pred_boot = (y_pred_prob_boot >= threshold).astype(int)
        
        # Calculate confusion matrix
        try:
            cm = confusion_matrix(y_true_boot, y_pred_boot)
            
            # Handle cases where confusion matrix might not be 2x2
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1, 1):
                # Only one class predicted
                failed_samples += 1
                continue
            else:
                failed_samples += 1
                continue
            
            # Calculate metrics with safer division
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
            
            bootstrap_sensitivities.append(sensitivity)
            bootstrap_specificities.append(specificity)
            bootstrap_ppvs.append(ppv)
            bootstrap_npvs.append(npv)
            bootstrap_f1s.append(f1)
            bootstrap_accuracies.append(accuracy)
            
        except (ValueError, IndexError) as e:
            failed_samples += 1
            continue
    
    print(f"Bootstrap completed. Failed samples: {failed_samples}/{n_bootstrap}")

    # Calculate confidence intervals (2.5th and 97.5th percentiles for 95% CI)
    def calculate_ci(values, alpha=0.05):
        if len(values) == 0:
            return np.nan, np.nan, np.nan
        sorted_values = np.sort(values)
        lower = np.percentile(sorted_values, 100 * alpha/2)
        upper = np.percentile(sorted_values, 100 * (1 - alpha/2))
        mean_val = np.mean(sorted_values)
        return mean_val, lower, upper
    
    results = {
        'auroc': calculate_ci(bootstrap_aurocs),
        'sensitivity': calculate_ci(bootstrap_sensitivities),
        'specificity': calculate_ci(bootstrap_specificities),
        'ppv': calculate_ci(bootstrap_ppvs),
        'npv': calculate_ci(bootstrap_npvs),
        'f1': calculate_ci(bootstrap_f1s),
        'accuracy': calculate_ci(bootstrap_accuracies),
        'n_successful_bootstraps': len(bootstrap_aurocs),
        'n_failed_bootstraps': failed_samples
    }
    
    return results

def format_ci_string(mean, lower, upper, decimals=3):
    """Format confidence interval as string"""
    # Handle NaN values
    if np.isnan(mean) or np.isnan(lower) or np.isnan(upper):
        return "N/A (insufficient data)"
    return f"{mean:.{decimals}f} ({lower:.{decimals}f}-{upper:.{decimals}f})"

def bootstrap_single_model(y_true, y_pred_prob, thresholds_dict, n_bootstrap=500, random_state=42):
    """
    Bootstrap analysis for a single model at multiple thresholds.
    """
    results_list = []
    
    print(f"Starting bootstrap analysis for {len(thresholds_dict)} thresholds...")
    
    for threshold_name, threshold_value in thresholds_dict.items():
        print(f"\nAnalyzing threshold: {threshold_name} ({threshold_value:.3f})")
        bootstrap_result = bootstrap_metrics(y_true, y_pred_prob, threshold_value, n_bootstrap, random_state)
        
        result_row = {
            'Threshold': f"{threshold_value:.3f} ({threshold_name})",
            'AUROC': format_ci_string(*bootstrap_result['auroc']),
            'Sensitivity': format_ci_string(*bootstrap_result['sensitivity']),
            'Specificity': format_ci_string(*bootstrap_result['specificity']),
            'PPV': format_ci_string(*bootstrap_result['ppv']),
            'NPV': format_ci_string(*bootstrap_result['npv']),
            'F1': format_ci_string(*bootstrap_result['f1']),
            'Accuracy': format_ci_string(*bootstrap_result['accuracy']),
            'Successful_Bootstraps': bootstrap_result['n_successful_bootstraps']
        }
        results_list.append(result_row)
    
    return pd.DataFrame(results_list)


# ====================================================================================
# calibration curve
# ====================================================================================

def plot_calibration_curve(y_true, y_pred_prob, n_bins=5, title="Calibration Plot", 
                          save_path=None, figsize=(8, 6)):
    """Plot calibration curve with both Brier Score and Log Loss."""
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_prob, n_bins=n_bins, strategy='uniform'
    )
    
    # Calculate both metrics
    brier_score = np.mean((y_pred_prob - y_true) ** 2)
    
    # Calculate Log Loss (avoid log(0) by clipping)
    eps = 1e-15  # Small value to avoid log(0)
    y_pred_clipped = np.clip(y_pred_prob, eps, 1 - eps)
    log_loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot calibration curve
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', 
             linewidth=2, markersize=8, 
             label=f'Brier Score: {brier_score:.3f}\nLog Loss: {log_loss:.3f}')
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', 
             label='Perfectly Calibrated', alpha=0.7)
    
    # # Add sample size information
    # total_samples = len(y_true)
    # plt.text(0.02, 0.98, f'Total samples: {total_samples}', 
    #          transform=plt.gca().transAxes, fontsize=10, 
    #          verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    plt.xlabel('Mean Predicted Probability', fontsize=14)
    plt.ylabel('Fraction of Positives', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return {
        'brier_score': brier_score,
        'log_loss': log_loss,
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value
    }