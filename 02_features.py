import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import gurobipy as gp
from utils import var_features, compute_epslabels

epsilon = 1e-6

parser = argparse.ArgumentParser()
parser.add_argument("-problem", type=int, default=0)
args = parser.parse_args()

base_dir = "./goodinstances/"
all_instances = os.listdir(base_dir)
for file in all_instances:
    if ".csv" in file:
        all_instances.remove(file)
all_instances.sort()


problem_name = all_instances[args.problem].split(sep=".")[0]

dir_name = "./goodfiles/" + problem_name + "/results/"

files = [file for file in os.listdir(dir_name) if "problem" not in file]
files.sort()

print(f"Read {len(files)} files for {problem_name}")

df_list = []
col_names = [
    "rounds",
    "lp_time",
    "sep_status",
    "sep_time",
    "num_cuts",
    "gap_closed",
    "sep_gap",
]
for file in files:
    data = pd.read_csv(
        dir_name + file,
        header=None,
        names=col_names,
    )
    df_list.append(data)

df_all = pd.concat([d.iloc[-1:] for d in df_list], ignore_index=True)

bad_files = df_all[df_all["gap_closed"] > 100]
bad_files = df_all[df_all["gap_closed"] < 0]
# how bad is this code?
print("Files with impossible gap:", len(bad_files))

# good_files = df_all[df_all["gap_closed"] <= 100]
good_files = df_all[df_all["gap_closed"] >= 5]
# print(len(good_files))


good_indx = good_files.index.tolist()
print("good files: ", len(good_indx))

unseen_files = df_all[df_all["gap_closed"] < 5]
unseen_idx = unseen_files.index.tolist()

directory = (
    # "/blue/akazachkov/o.guaje/" + "gooddata/" + problem_name + "/"
    "./goodfiles/"
    + problem_name
    + "/"
)

data_dir = directory + "data/"
try:
    os.makedirs(data_dir, exist_ok=True)
except FileExistsError:
    pass

with open(directory + "unseen.txt", "w") as f:
    f.write(str(unseen_idx))
# good_indx.remove(810)
for my_idx in good_indx:

    ya_existe = os.path.isfile(data_dir + str(my_idx).zfill(4) + ".csv")

    if not ya_existe:
        read_dir = directory + "random/"

        alphas_dir = directory + "cuts/" + str(my_idx).zfill(4) + "/"

        sols_dir = directory + "sols/" + str(my_idx).zfill(4) + "/"

        multipliers_dir = (
            directory + "full_lambdas/" + str(my_idx).zfill(4) + "/"
        )

        ip = gp.read(read_dir + str(my_idx).zfill(4) + ".mps")
        var_types = ip.getAttr("VType")
        bounds = ip.getAttr("UB")

        for i in range(len(bounds)):
            if bounds[i] == 1 and var_types[i] == "I":
                var_types[i] = "B"

        num_rounds = len(os.listdir(alphas_dir))

        lp = ip.relax()
        lp.optimize()

        solution = np.array([v.x for v in lp.getVars()])

        all_data = []

        for i in range(num_rounds):
            this_dataset, feature_names = var_features(lp, solution, var_types)
            this_dataset = pd.DataFrame(this_dataset, columns=feature_names)
            labels = np.loadtxt(
                alphas_dir + str(i) + ".csv", delimiter=","
            )
            if len(labels.shape) == 1:
                labels = labels.reshape(1, -1)
            if labels.shape[1] == 0:
                break
            labels = labels[:, : len(lp.getVars())]
            iter_label = [i] * this_dataset.shape[0]
            instance_label = [my_idx] * this_dataset.shape[0]
            label_1 = [
                1 if len(np.nonzero(labels[:, i])[0]) else 0
                for i in range(labels.shape[1])
            ]
            label_2 = compute_epslabels(labels, 0.1)
            label_3 = np.max(np.abs(labels), axis=0)
            this_dataset["instance_id"] = instance_label
            this_dataset["cut_iter"] = iter_label
            this_dataset["label_any"] = label_1
            this_dataset["label_eps"] = label_2
            this_dataset["label_max"] = label_3
            all_data.append(this_dataset)
            solution = np.loadtxt(sols_dir + str(i) + ".csv", delimiter=",")

        huge_dataset = pd.concat(all_data)

        huge_dataset.to_csv(
            data_dir + str(my_idx).zfill(4) + ".csv", index=False
        )
