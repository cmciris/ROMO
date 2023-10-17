import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker


def hartmann(x0, x1, x2):
    n_samples = x0.shape[0]
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = np.array([[3689, 1170, 2673],
                  [4699, 4387, 7470],
                  [1091, 8732, 5547],
                  [381, 5743, 8828]]) * 1e-4
    x = np.array([[x0, x1, x2]])
    x = np.transpose(x, [2,0,1])
    x = np.repeat(x, repeats=4, axis=1)
    P = np.repeat(np.array([P]), repeats=n_samples, axis=0)
    exponent = np.einsum("ijk,jk->ij", (x - P) ** 2, A)
    y = -np.einsum("ij,j->i", np.exp(-exponent), alpha)
    return -y


def main(path, stride=5, samples=15, fontsize=5, sublayer=[4,5]):
    os.makedirs(path, exist_ok=True)

    if os.getcwd().endswith("/visualization"):
        logs_dir = os.path.join("../logs", path)
        x = np.load("../data/x.npy")
    else:
        logs_dir = os.path.join("./logs", path)
        x = np.load("./data/x.npy")

    x0 = x[:,0]
    x1 = x[:,1]
    x2 = x[:,2]
    y = hartmann(x0, x1, x2)

    argsort = np.argsort(y)
    indices = argsort[1000:-1000]
    x0 = x0[indices]
    x1 = x1[indices]
    x2 = x2[indices]

    for root, _, files in os.walk(logs_dir):
        if root == logs_dir: break

    for file in files:
        if file.endswith("_solutions.npy"):
            solutions = np.load(os.path.join(logs_dir, file))
            steps = solutions.shape[1]

            fig, axes = plt.subplots(nrows=sublayer[0], ncols=sublayer[1], figsize=(6, 5))
            tmp = plt.get_cmap('autumn_r')(np.linspace(0, 1, 7))
            colors = [(0, tmp[1]), (0.05, tmp[2]), (0.1, tmp[3]), (0.2, tmp[4]), (1, tmp[5])]
            cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
            for i in range(sublayer[0]):
                for j in range(sublayer[1]):
                    layer = i * sublayer[1] + j
                    ax = axes[i, j]
                    ax.tick_params(axis='both', labelsize=fontsize)
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position('top')
                    if i != 0:
                        ax.set_xticks([])
                    else:
                        ax.set_xticks([0.0, 0.5, 1.0])
                        ax.set_xlabel("x0", fontsize=fontsize)
                    if j != 0:
                        ax.set_yticks([])
                    else:
                        ax.set_yticks([0.0, 0.5, 1.0])
                        ax.set_ylabel("x1", fontsize=fontsize)
                    ax.set_xlim(-0.035, 1.03)
                    ax.set_ylim(-0.03, 1.03)
                    ax.set_title("{} < x2 < {}".format(layer / np.prod(sublayer), (layer + 1) / np.prod(sublayer)), fontsize=fontsize+1, y=-0.18)

                    indices = [(x2 > layer / np.prod(sublayer)) & (x2 < (layer + 1) / np.prod(sublayer))][0]
                    tmp_x0 = x0[indices]
                    tmp_x1 = x1[indices]
                    tmp_x2 = x2[indices]
                    y = hartmann(tmp_x0, tmp_x1, tmp_x2)
                    ax.scatter(tmp_x0, tmp_x1, s=0.5, zorder=1)

                    tmp = solutions[((solutions[:,:,2] < (layer + 1) / np.prod(sublayer)) & (solutions[:,:,2] > layer / np.prod(sublayer)))[:,0]]
                    for point in tmp:
                        ax.plot(point[0:steps,0], point[0:steps,1], c="k", linewidth=0.1, zorder=2)
                    for point in tmp:
                        scatter = ax.scatter(point[0:steps,0], point[0:steps,1], c=np.arange(51) / 50, cmap=cmap, s=0.5, zorder=3)
                    
            figname = os.path.join(path, "boost_" + file[:-14] + ".pdf")
            plt.tight_layout(pad=0.4)
            plt.savefig(figname, dpi=1200)
            plt.close("all")


if __name__ == "__main__":
    main(path="gradient_ascent", stride=1)
    main(path="coms_cleaned", stride=1)
    main(path="iom", stride=1)
    main(path="rim_ridge", stride=1)
    main(path="romo_ridge", stride=1)