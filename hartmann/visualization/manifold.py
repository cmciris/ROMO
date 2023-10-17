import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def main(out_dir="./", samples=15, fontsize=12,
         x2_range = [[0.35, 0.4], [0.65, 0.7], [0.9, 0.95]]):
    sublayer = [len(x2_range), 6]

    grad_id = "0903130724"
    grad_x = np.load(os.path.join("../logs/gradient_ascent/", grad_id + "_pred_x.npy"))
    grad_y = np.load(os.path.join("../logs/gradient_ascent/", grad_id + "_pred_y.npy"))
    grad_xy = np.concatenate([grad_x, grad_y], axis=-1)
    coms_id = "0903145437"
    coms_x = np.load(os.path.join("../logs/coms_cleaned/", coms_id + "_pred_x.npy"))
    coms_y = np.load(os.path.join("../logs/coms_cleaned/", coms_id + "_pred_y.npy"))
    coms_xy = np.concatenate([coms_x, coms_y], axis=-1)
    iom_id = "0903132148"
    iom_x = np.load(os.path.join("../logs/iom/", iom_id + "_pred_x.npy"))
    iom_y = np.load(os.path.join("../logs/iom/", iom_id + "_pred_y.npy"))
    iom_xy = np.concatenate([iom_x, iom_y], axis=-1)
    romo_id = "0912043548"
    romo_x = np.load(os.path.join("../logs/romo_ridge/", romo_id + "_pred_x.npy"))
    romo_y = np.load(os.path.join("../logs/romo_ridge/", romo_id + "_pred_y.npy"))
    romo_xy = np.concatenate([romo_x, romo_y], axis=-1)
    rim_id = "0912042643"
    rim_x = np.load(os.path.join("../logs/rim_ridge/", rim_id + "_pred_x.npy"))
    rim_y = np.load(os.path.join("../logs/rim_ridge/", rim_id + "_pred_y.npy"))
    rim_xy = np.concatenate([rim_x, rim_y], axis=-1)
    models = [grad_xy, coms_xy, iom_xy, romo_xy, rim_xy]
    models_name = ["Grad. Ascent", "COMs", "IOM", "ROMO", "REM"]

    x0 = np.linspace(0, 1, samples)
    x1 = np.linspace(0, 1, samples)
    x0, x1 = np.meshgrid(x0, x1)
    np.random.seed(42)
    random = np.random.random(samples * samples)

    fig, axes = plt.subplots(nrows=sublayer[0], ncols=sublayer[1], figsize=(9,4.5))
    for i in range(sublayer[0]):
        for j, pred_xy in enumerate(models):
            tmp = pred_xy[(pred_xy[:,2] > x2_range[i][0]) & (pred_xy[:,2] < x2_range[i][1])]
            df = pd.DataFrame(tmp)
            tmp = df.sort_values([1, 0]).values
            pred_x0 = tmp[:,0].reshape(samples, samples)
            pred_x1 = tmp[:,1].reshape(samples, samples)
            pred_y = tmp[:,-1].reshape(samples, samples)

            ax = axes[i, j]
            if i == 0:
                ax.set_title(models_name[j], fontsize=fontsize + 3)
            if i != sublayer[0] - 1:
                ax.set_xticks([])
            else:
                ax.set_xticks([])
                # ax.set_xticks([0.0, 0.5, 1.0])
                # ax.set_xlabel("x0", fontsize=fontsize + 2)
            if j != 0:
                ax.set_yticks([])
            else:
                ax.set_yticks([])
                # ax.set_yticks([0.0, 0.5, 1.0])
                # ax.set_ylabel("({} < x2 < {})\nx1".format(x2_range[i][0], x2_range[i][1]), fontsize=fontsize + 2)
            ax.tick_params(axis='both', labelsize=fontsize)
            contour = ax.contourf(pred_x0, pred_x1, pred_y, levels=20, cmap="Blues")
            # cbar = fig.colorbar(contour, ax=ax)
            # cbar.ax.tick_params(labelsize=fontsize)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    for i in range(sublayer[0]):
        x2 = x2_range[i][0] + random * (x2_range[i][1] - x2_range[i][0])
        y = hartmann(x0.reshape(-1), x1.reshape(-1), x2).reshape(samples, samples)

        ax = axes[i, -1]
        if i == 0:
            ax.set_title("Truth", fontsize=fontsize + 3)
        if i != sublayer[0] - 1:
            ax.set_xticks([])
        else:
            ax.set_xticks([])
            # ax.set_xticks([0.0, 0.5, 1.0])
            # ax.set_xlabel("x0", fontsize=fontsize + 2)
        ax.set_yticks([])
        ax.tick_params(axis='both', labelsize=fontsize)
        contour = ax.contourf(x0, x1, y, levels=20, cmap="Blues")
        # cbar = fig.colorbar(contour, ax=ax)
        # cbar.ax.tick_params(labelsize=fontsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    figname = os.path.join(out_dir, "manifold.pdf")
    plt.tight_layout(pad=0.4)
    plt.savefig(figname, dpi=1200)

if __name__ == "__main__":
    main(out_dir="./")