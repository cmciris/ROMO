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


def main(path, stride=5, samples=15, fontsize=10, sublayer=[1,3],
         x2_range = [[0.35, 0.4], [0.65, 0.7], [0.9, 0.95]]):
    os.makedirs(path, exist_ok=True)

    if os.getcwd().endswith("/visualization"):
        logs_dir = os.path.join("../logs", path)
    else:
        logs_dir = os.path.join("./logs", path)

    x0 = np.linspace(0, 1, samples)
    x1 = np.linspace(0, 1, samples)
    x0, x1 = np.meshgrid(x0, x1)
    np.random.seed(42)
    random = np.random.random(samples * samples)

    for root, _, files in os.walk(logs_dir):
        if root == logs_dir: break

    for file in files:
        if file.endswith("_solutions.npy"):
            solutions = np.load(os.path.join(logs_dir, file))
            steps = solutions.shape[1]

            fig, axes = plt.subplots(nrows=sublayer[0], ncols=sublayer[1], figsize=(6, 1.8))
            tmp = plt.get_cmap('autumn_r')(np.linspace(0, 1, 7))
            colors = [(0, tmp[1]), (0.05, tmp[2]), (0.1, tmp[3]), (0.2, tmp[4]), (1, tmp[5])]
            cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
            for i in range(sublayer[0]):
                for j in range(sublayer[1]):
                    layer = i * sublayer[1] + j
                    if sublayer[0] == 1:
                        ax = axes[j]
                    else:
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
                    ax.set_xlim(-0.02, 1.02)
                    ax.set_ylim(-0.03, 1.03)
                    # ax.set_xlim(0, 1)
                    # ax.set_ylim(0, 1)
                    ax.set_title("{} < x2 < {}".format(x2_range[layer][0], x2_range[layer][1]), fontsize=fontsize+1, y=-0.24)

                    x2 = x2_range[layer][0] + random * (x2_range[layer][1] - x2_range[layer][0])
                    y = hartmann(x0.reshape(-1), x1.reshape(-1), x2).reshape(samples, samples)
                    contour = ax.contourf(x0, x1, y, levels=20, cmap="Blues")
                    # cbar = fig.colorbar(contour, ax=ax)
                    # cbar.ax.tick_params(labelsize=fontsize)

                    tmp = solutions[((solutions[:,:,2] < x2_range[layer][1]) & (solutions[:,:,2] > x2_range[layer][0]))[:,0]]
                    for point in tmp:
                        ax.plot(point[0:steps:stride,0], point[0:steps:stride,1], c="k", linewidth=0.1, zorder=1)
                    for point in tmp:
                        scatter = ax.scatter(point[0:steps:stride,0], point[0:steps:stride,1], c=np.arange(51) / 50, cmap=cmap, s=8, zorder=2, alpha=0.5, marker='o', edgecolor='k', linewidth=0.1)
                        scatter = ax.scatter(point[0:steps:stride,0], point[0:steps:stride,1], c=np.arange(51) / 50, cmap=cmap, s=3, zorder=3, alpha=0.5, marker='o')

                    # cbar = fig.colorbar(scatter, ticks=np.linspace(0, 1, 6), ax=ax, pad=0.06)
                    # cbar.ax.tick_params(labelsize=fontsize)
                    # cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*50:.0f}'))

            figname = os.path.join(path, "track_" + file[:-14] + ".pdf")
            plt.tight_layout(pad=0.5)
            plt.savefig(figname, dpi=1200)
            plt.close("all")


if __name__ == "__main__":
    main(path="gradient_ascent", stride=1)
    main(path="coms_cleaned", stride=1)
    main(path="iom", stride=1)
    main(path="rim_ridge", stride=1)
    main(path="romo_ridge", stride=1)