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

    sns.set_style('whitegrid')
    y_data = np.squeeze(y_data)
    y_oracle = np.squeeze(y_oracle)
    sns.scatterplot(x=y_data, y=y_oracle, zorder=1, alpha=1, label='$(y_\mathcal{D}, f(\mathbf{x}))$')
    plt.plot([-4,4], [-4,4], c='orangered', zorder=2, alpha=0.8, label='$f(\mathbf{x})=y_\mathcal{D}$', linewidth=1.5, linestyle='-')
    plt.xlabel("$y_\mathcal{D}$")
    plt.ylabel("$f(\mathbf{x})$")
    plt.legend(loc='lower right')
    plt.savefig('./hopper_fig/scatter.png')
    plt.savefig('./hopper_fig/scatter.pdf')

if __name__ == "__main__":

    main()