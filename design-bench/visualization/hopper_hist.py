import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pdb

def main():
    normalize = True
    if normalize:
        # x = np.load("./hopper_normalized_x.npy")
        y_data = np.load("./hopper_normalized_y_data.npy")
        y_oracle = np.load("./hopper_normalized_y_oracle.npy")
    else:
        # x = np.load("./hopper_unnormalized_x.npy")
        y_data = np.load("./hopper_unnormalized_y_data.npy")
        y_oracle = np.load("./hopper_unnormalized_y_oracle.npy")
    print(y_data.shape, y_oracle.shape)

    
    y_data = np.squeeze(y_data)
    y_oracle = np.squeeze(y_oracle)
    sns.set_style('white')
    sns.histplot(y_data, label='$y_{\mathcal{D}}$', zorder=2, alpha=0.5)
    sns.histplot(y_oracle, label='$f(\mathbf{x})$', zorder=1, alpha=0.5)
    if normalize:
        plt.xlabel("Normalized Score")
    else:
        plt.xlabel("Unnormalized Score")
    plt.legend(loc='lower right')
    plt.savefig('./hopper_fig/hist.png')
    plt.savefig('./hopper_fig/hist.pdf')

if __name__ == "__main__":
    main()