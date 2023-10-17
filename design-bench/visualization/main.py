import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pdb

def main():
    file_dir = '../logs/gradient_ascent'
    run_id = '0625143224'
    auxiliaries = np.load(os.path.join(file_dir, f'{run_id}_auxiliaries.npy'))  # 128, N, 80
    predictions = np.load(os.path.join(file_dir, f'{run_id}_predictions.npy'))  # 128, N, 1
    scores = np.load(os.path.join(file_dir, f'{run_id}_scores.npy'))  # 128, N, 1
    solutions = np.load(os.path.join(file_dir, f'{run_id}_solutions.npy'))  # 128, N, 3640
    solution = np.load(os.path.join(file_dir, f'{run_id}_solution.npy'))  # 128, 3640
    
    assert np.all(solutions[:, -1, :] == solution)

    n_samples = auxiliaries.shape[0]
    n_steps = auxiliaries.shape[1]
    n_stations = auxiliaries.shape[2] // 2
    auxiliaries = auxiliaries.reshape([n_samples, n_steps, n_stations, 2])
    
    scope_id = 1
    for step in range(n_steps):
        cio = solutions[scope_id, step, :n_stations*n_stations]
        # cio_df = pd.DataFrame(cio.reshape([n_stations, n_stations])).iloc[::-1]
        cio_df = pd.DataFrame(cio.reshape([n_stations, n_stations]))
        cio_heatmap = sns.heatmap(data=cio_df, square=True, vmin=-6.0, vmax=6.0, cmap="RdBu_r")
        heatmap = cio_heatmap.get_figure()
        heatmap.savefig(f'./figs/{step+1}_cio.jpg', dpi=400)
        plt.close()
        # cnt = 0
        # for i in cio:
        #     if i > 4.0 or i < -4.0:
        #         cnt += 1
        # print(cnt)

        print(step+1)
        
        # if step == n_steps-1 or step == 0:
        aux = auxiliaries[scope_id, step, :, :2]

        x = np.arange(auxiliaries.shape[2])
        prb_up = aux[..., 0]
        prb_down = aux[..., 1]

        total_width, n = 0.8, aux.shape[-1]
        width = total_width / n
        x = x - (total_width - width) / 2

        plt.bar(x, prb_up, width=width, label='prb_up')
        plt.bar(x + width, prb_down, width=width, label='prb_down')
        plt.ylim((0, 100))
        # plt.xticks([])
        plt.savefig(f'./figs/{step+1}.jpg')
        plt.cla()
        plt.close()
        
        

if __name__ == "__main__":
    main()
