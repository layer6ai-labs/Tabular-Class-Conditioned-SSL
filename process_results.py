import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind
from utils import *

ALL_DIDS = [11, 14, 15, 16, 18, 22, 
            23, 29, 31, 37, 50, 54, 
            188, 458, 469, 1049, 1050, 1063, 
            1068, 1462, 1464, 1480, 1494, 1510,    
            6332, 23381, 40966, 40975, 40982, 40994]

METHODS_TO_COMPARE = ['no_pretrain', 'rand_corr-rand_feats', 'cls_corr-rand_feats', 'orc_corr-rand_feats']
# METHODS_TO_COMPARE = ['cls_corr-rand_feats', 'cls_corr-leastRela_feats', 'cls_corr-mostRela_feats']

print(f"Process results for {METHODS_TO_COMPARE} on the metric {METRIC}")

if __name__ == "__main__":
    win_mat = np.zeros(shape=[len(METHODS_TO_COMPARE), len(METHODS_TO_COMPARE)])
    datasets_list = openml.datasets.list_datasets(ALL_DIDS, output_format='dataframe')
    if 'cls_corr-leastRela_feats' not in METHODS_TO_COMPARE:
        # Table including comparison between methods on how to corrupt 
        latex_table = "\hline \n Datasets (DID) & No-PreTrain & Random & Class & Oracle \\\\ \n\hline \n"
    else:
        # Table including comparison between methods on where to corrupt
        latex_table = "\hline \n Datasets (DID) & \makecell{Feature Correlation \\\\ Value Range} & Random Features & Least Correlated & Most Correlated \\\\ \n\hline \n"

    for did in ALL_DIDS:
        if not os.path.isdir(os.path.join(RESULT_DIR, f"DID_{did}")):
            print(f"Dataset {did} results not available! Skipped.")
            continue
        res_vals, res_avg, res_std = {}, {}, {}
        for method in METHODS_TO_COMPARE:
            if METRIC == "accuracy":
                res_vals[method] = np.load(os.path.join(RESULT_DIR, f"DID_{did}", f"{method}_accuracies.npy"))
            else:
                res_vals[method] = np.load(os.path.join(RESULT_DIR, f"DID_{did}", f"{method}_aurocs.npy"))
            assert len(res_vals[method]) == len(SEEDS)
            res_avg[method] = np.mean(res_vals[method])
            res_std[method] = np.std(res_vals[method]) / np.sqrt(len(SEEDS))

        if 'cls_corr-leastRela_feats' in METHODS_TO_COMPARE:
            # Read in feature correlation value range
            spec_file = os.path.join(RESULT_DIR, f"DID_{did}", "experimentSpecs.txt")
            with open(spec_file, "r") as res_f: 
                spec_lines = res_f.readlines()
                val_token = spec_lines[2].split(' ')[-1].rstrip()
                try:
                    feature_correlation_value_range = float(val_token)
                except ValueError:
                    print(f"Invalid string for correlation value range: {val_token}")
        else:
            feature_correlation_value_range = None

        # Update the win matrix
        for i in range(len(METHODS_TO_COMPARE)):
            for j in range(i+1, len(METHODS_TO_COMPARE)):
                method_1, method_2 = METHODS_TO_COMPARE[i], METHODS_TO_COMPARE[j]
                # conduct Welch's t-test with unequal variances
                if res_avg[method_1] < res_avg[method_2]:
                    # Null hypothesis to be rejected: method_1 has higher mean
                    t_stat, p_val = ttest_ind(a=res_vals[method_1], 
                                              b=res_vals[method_2], 
                                              equal_var=False, 
                                              alternative='less')
                    if p_val < P_VAL_SIGNIFICANCE:
                        # Null hypothesis rejected, method_2 has higher mean
                        win_mat[j][i] += 1
                else:
                    # Null hypothesis to be rejected: method_2 has higher mean
                    t_stat, p_val = ttest_ind(a=res_vals[method_2], 
                                              b=res_vals[method_1], 
                                              equal_var=False, 
                                              alternative='less')
                    if p_val < P_VAL_SIGNIFICANCE:
                        # Null hypothesis rejected, method_1 has higher mean
                        win_mat[i][j] += 1

        # Write avg and std statistics in latex
        ds_name = datasets_list[datasets_list.did==did].name.item()
        ds_name = ds_name.replace("_", "-")
        if 'cls_corr-leastRela_feats' not in METHODS_TO_COMPARE:
            latex_table += f"{ds_name} ({did}) & " 
            # add in results under random features selected for corruption
            latex_table += f"${res_avg['no_pretrain']:.2f}\pm {res_std['no_pretrain']:.2f}$ & "
            latex_table += f"${res_avg['rand_corr-rand_feats']:.2f}\pm {res_std['rand_corr-rand_feats']:.2f}$ & "
            latex_table += f"${res_avg['cls_corr-rand_feats']:.2f}\pm {res_std['cls_corr-rand_feats']:.2f}$ & "
            latex_table += f"${res_avg['orc_corr-rand_feats']:.2f}\pm {res_std['orc_corr-rand_feats']:.2f}$ \\\\ \n"
        else:
            latex_table += f"{ds_name} ({did}) & {feature_correlation_value_range} & " 
            # add in results under class-conditioned corruption
            latex_table += f"${res_avg['cls_corr-rand_feats']:.2f}\pm {res_std['cls_corr-rand_feats']:.2f}$ & "
            latex_table += f"${res_avg['cls_corr-leastRela_feats']:.2f}\pm {res_std['cls_corr-leastRela_feats']:.2f}$ & "
            latex_table += f"${res_avg['cls_corr-mostRela_feats']:.2f}\pm {res_std['cls_corr-mostRela_feats']:.2f}$ \\\\ \n"
      

    # process win matrices
    win_mat_divisor = win_mat + np.transpose(win_mat) + np.eye(len(METHODS_TO_COMPARE))
    win_mat = np.divide(win_mat, win_mat_divisor)
    print(f"Win matrix for {METRIC}: \n", win_mat)

    fig, ax = plt.subplots()
    im = ax.imshow(win_mat, cmap='autumn', alpha=0.6)

    ax.set_xticks(np.arange(len(METHODS_TO_COMPARE)))
    ax.set_yticks(np.arange(len(METHODS_TO_COMPARE)))
    if 'cls_corr-leastRela_feats' not in METHODS_TO_COMPARE:
        assert len(METHODS_TO_COMPARE) == 4
        ax.set_xticklabels(['No Pre-Train', 'Conventional', 'Class-Conditioned', 'Oracle'])
        ax.set_yticklabels(['No Pre-Train', 'Conventional', 'Class-Conditioned', 'Oracle'])
    else:
        assert len(METHODS_TO_COMPARE) == 3
        ax.set_xticklabels(['Conventional', 'Least Correlated Features', 'Most Correlated Features'])
        ax.set_yticklabels(['Conventional', 'Least Correlated Features', 'Most Correlated Features'])
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(METHODS_TO_COMPARE)):
        for j in range(len(METHODS_TO_COMPARE)):
            text = ax.text(j, i, f"{win_mat[i, j]*100:.1f}%",
                        ha="center", va="center", color="k", weight="bold", size=23)
    fig.tight_layout()
    plt.show()

    # write the results to a file
    # finishing the table by an underline
    latex_table += "\hline \n"  
    latex_table_filename = os.path.join(RESULT_DIR, 
                                        f"{METRIC}_table_{'where_corrupt' if 'cls_corr-leastRela_feats' in METHODS_TO_COMPARE else 'how_corrupt'}.tex")

    with open(latex_table_filename, "w") as f:
        f.write(latex_table)
    
    print(f"Latex table generated and saved to {latex_table_filename} file!")

    print("Script finished!")