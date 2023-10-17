import os
import argparse
import numpy as np
import re

def main(path, detail=False, std=False, start_step=240, end_step=250):
    if os.getcwd().endswith("/visualization"):
        logs_dir = os.path.join("../logs", path)
    else:
        logs_dir = os.path.join("./logs", path)

    for root, _, files in os.walk(logs_dir):
        if root == logs_dir: break
    
    seeds = []
    for file in files:
        if file.endswith("_scores.npy"):
            one_seed = []
            with open(os.path.join(logs_dir, file[:-11] + "_info.log"), "r") as info:
                while True:
                    line = info.readline().strip()
                    if line.startswith("Epoch ["):
                        epochs = re.split('\D+', line.strip())
                        assert len(epochs) == 4
                        if epochs[1] == epochs[2]:
                            break
                while True:
                    line = info.readline().strip()
                    if (line.startswith("(train/nll") and "mean" in line) or (line.startswith("(train/mse") and "mean" in line):
                        one_seed.append(float(line.split(",")[0].split(":")[-1]))
                        break
                while True:
                    line = info.readline().strip()
                    if (line.startswith("(validate/nll") and ("mean" in line or "value" in line)) or (line.startswith("(validate/mse") and ("mean" in line or "value" in line)):
                        one_seed.append(float(line.split(",")[0].split(":")[-1]))
                        break
            scores = np.load(os.path.join(logs_dir, file))

            n_particles, steps = scores.shape[0], scores.shape[1]
            scores = scores.reshape(n_particles, steps)[:,start_step:end_step]
            try:
                best = np.sort(scores, axis=-1)[:,-1]
            except:
                continue

            one_seed.append(best.mean())
            for quantile in [1, 0.9, 0.8, 0.5]:
                one_seed.append(np.quantile(best, q=quantile))
            seeds.append(one_seed)
    seeds = np.stack(seeds)
    if detail:
        for seed in seeds:
            for value in seed:
                print("%.3f" % value, end=", ")
            print("")
        print("Average: ", end="")
    if std:
        for index in range(seeds.shape[1]):
            print("%.3f" % seeds[:, index].mean() + "±" + "%.3f" % seeds[:, index].std(), end=", ")
    else:
        for index in range(seeds.shape[1]):
            print("%.3f" % seeds[:, index].mean(), end=", ")
    print("\t-{}".format(path), end="\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", default=["gradient_ascent", "coms", "iom", "rim_ridge", "romo_ridge"])
    parser.add_argument("-v", "--env", default="hopper")
    args = parser.parse_args()
    if type(args.exp) == str: args.exp = [args.exp]
    if type(args.env) == str: args.env = [args.env]
    for env in args.env:
        for exp in args.exp:
            main(path=os.path.join(exp, env), detail=False, std=True, start_step=40, end_step=50)