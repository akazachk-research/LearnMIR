import sys
import os
import copy
import time
import argparse
import numpy as np
import pandas as pd
import gurobipy as gp
import joblib as jb
from sklearn.preprocessing import StandardScaler
from mirsep import Mirsep
from utils import features

parser = argparse.ArgumentParser()
parser.add_argument("-directory", "--directory", type=str, default="")
parser.add_argument("-problem", "--problem", type=str, default="")
parser.add_argument("-index", "--index", "-i", type=int, default=0)
parser.add_argument("-rounds", "--rounds", "-r", type=int, default=0)
parser.add_argument("-outdir", "--outdir", "-out_directory", "--out_directory", type=str, default="results")
args = parser.parse_args()

instance_idx = args.index

# base_dir = "/blue/akazachkov/o.guaje/goodinstances/"
base_dir = "./goodinstances/"
all_instances = os.listdir(base_dir)
for file in all_instances:
    if ".csv" in file:
        all_instances.remove(file)
all_instances.sort()

# problem_name = all_instances[args.problem].split(sep=".")[0]
problem_name = args.problem
if problem_name == "":
    print("Please provide a problem name")
    sys.exit(0)
directory_name = args.directory
if directory_name == "":
    print("Please provide a directory name where the data is located")
    sys.exit(0)

in_directory = (
    # "/blue/akazachkov/o.guaje/" + "gooddata/" + problem_name + "/"
    directory_name
    + "/"
    + problem_name
    + "/"
)

out_directory = (
    args.outdir + "/" + problem_name + "/"
)

# Process maximum number of rounds
max_rounds = args.rounds
if max_rounds <= 0:
    max_rounds = 1000

opt_threshold = np.load(
    "./models/" + problem_name + "/optimal_threshold_" + problem_name + ".npy"
)
opt_threshold = opt_threshold[0]

learned_model = jb.load(
    "./models/"
    + problem_name
    + "/GB_classifier_model___"
    + problem_name
    + ".joblib"
)

read_dir = in_directory + "random/"

alphas_dir = out_directory + "reduced_cuts/" + str(instance_idx).zfill(4) + "/"
try:
    os.makedirs(alphas_dir, exist_ok=True)
except FileExistsError:
    pass

sols_dir = out_directory + "reduced_sols/" + str(instance_idx).zfill(4) + "/"
try:
    os.makedirs(sols_dir, exist_ok=True)
except FileExistsError:
    pass

multipliers_dir = (
    out_directory + "reduced_lambdas/" + str(instance_idx).zfill(4) + "/"
)
try:
    os.makedirs(multipliers_dir, exist_ok=True)
except FileExistsError:
    pass

logs_dir = out_directory + "reduced_logs/" + str(instance_idx).zfill(4) + "/"
try:
    os.makedirs(logs_dir, exist_ok=True)
except FileExistsError:
    pass


# instance = all_instances[args.problem]
# instance_id = instance.split(sep=".")[0]
instance_id = problem_name

ip = gp.read(read_dir + str(instance_idx).zfill(4) + ".mps")

ip.Params.OutputFlag = 0
ip.Params.LogFile = ""
ip.Params.TimeLimit = 3600 * 0.5

print("Solving IP")
ip.optimize()
print("Solved IP")
print(ip.Runtime)

if ip.Status != 2:
    # print(problem_name, instance, " broke on IP solve with status ", ip.Status)
    print(problem_name, " broke on IP solve with status ", ip.Status)
    sys.exit(0)


ip_val = ip.ObjVal

int_solution = [v.x for v in ip.getVars()]
var_types = ip.getAttr("VType")

lp = ip.relax()

print("Solving relaxation")
lp.optimize()
print("Solved relaxation")

lp_base = lp.ObjVal
solution = [v.x for v in lp.getVars()]
old_solution = copy.deepcopy(solution)


logfile = logs_dir + instance_id + "_"

# results_dir = "/blue/akazachkov/o.guaje/results/"
results_dir = out_directory + "reduced_results/"
try:
    os.mkdir(results_dir)
except FileExistsError:
    pass

instance_name = str(instance_idx).zfill(4)
outputfile = results_dir + instance_name + ".txt"
problemfile = results_dir + "problems_" + instance_name + ".txt"

rounds = 0
continuar = True
tic = time.time()
nic = time.process_time()
separator = Mirsep(ip, solution, 5, 600)
toc = time.time()
noc = time.process_time()

print("created separator in ", toc - tic, "wall seconds")
print("created separator in ", noc - nic, "cpu seconds")

this_dataset, feature_names = features(lp, np.array(solution), var_types)
this_dataset = pd.DataFrame(this_dataset, columns=feature_names)
this_dataset["instance_id"] = [instance_idx] * this_dataset.shape[0]
this_dataset["cut_iter"] = [rounds] * this_dataset.shape[0]

scaler = StandardScaler()
this_dataset = scaler.fit_transform(this_dataset)

predictions = learned_model.predict_proba(this_dataset)
predictions = [1 if p[1] > opt_threshold else 0 for p in predictions]

while continuar:
    print("Starting separation round ", rounds)
    tic = time.time()
    nic = time.process_time()
    separator.build_model(
        logs_dir + str(rounds).zfill(4) + ".log", predictions
    )
    toc = time.time()
    noc = time.process_time()
    print("built model in ", toc - tic, "wall seconds")
    print("built model in ", noc - nic, "cpu seconds")
    separator.solve()
    print("Finished separation")

    if separator.model.status not in [2, 9, 11]:
        print(
            problem_name,
            instance,
            " broke in separation with status ",
            separator.model.status,
        )
        sys.exit()

    cuts = separator.get_cuts()
    lambdas = separator.get_lambdas()

    num_cuts = len(cuts)

    valid_cuts = []

    variables = lp.getVars()
    tic = time.time()
    nic = time.process_time()
    for i in range(num_cuts):
        if (
            np.sum(
                [
                    cuts[i][j] * int_solution[j]
                    for j in range(len(int_solution))
                ]
            )
            < cuts[i][-1]
        ):
            line = str(
                np.sum(
                    [
                        cuts[i][j] * int_solution[j]
                        for j in range(len(int_solution))
                    ]
                )
            )
            line = line + " " + str(cuts[i][-1])
            with open(problemfile, "a") as f:
                f.write(
                    "invalid cut"
                    + str(i)
                    + " in round "
                    + str(rounds)
                    + " "
                    + line
                    + "\n"
                )
        else:
            valid_cuts.append(i)
            lp.addConstr(
                gp.quicksum(
                    [
                        cuts[i][j] * variables[j]
                        for j in range(len(int_solution))
                    ]
                )
                >= cuts[i][-1],
                name="cgmipcut",
            )

    toc = time.time()
    noc = time.process_time()
    print("added cuts in ", toc - tic, "wall seconds")
    print("added cuts in ", noc - nic, "cpu seconds")

    lp.update()
    lp.optimize()

    lp_solution = [v.x for v in lp.getVars()]
    pd.DataFrame(lp_solution).to_csv(
        sols_dir + str(rounds) + ".csv",
        index=False,
        header=False,
    )

    # active_cuts = []
    # for i in valid_cuts:
    #     if (
    #         abs(np.sum(
    #             [
    #                 cuts[i][j] * int_solution[j]
    #                 for j in range(len(int_solution))
    #             ]
    #         ) - cuts[i][-1]) < 1e-6
    #     ):
    #         active_cuts.append(i)

    # pd.DataFrame([lambdas[i] for i in active_cuts]).to_csv(
    pd.DataFrame([lambdas[i] for i in valid_cuts]).to_csv(
        multipliers_dir + str(rounds) + ".csv",
        index=False,
        header=False,
    )
    pd.DataFrame([cuts[i] for i in valid_cuts]).to_csv(
        alphas_dir + str(rounds) + ".csv",
        index=False,
        header=False,
    )
    lp_val_all = lp.ObjVal
    if abs(ip_val - lp_base) > 1e-6:
        gap_closed_all = 100 - 100 * (
            (ip_val - lp_val_all) / (ip_val - lp_base)
        )
    else:
        gap_closed_all = -1

    line = (
        str(rounds)
        + ", "
        + str(lp.Runtime)
        + ", "
        + str(separator.model.status)
        + ", "
        + str(separator.model.Runtime)
        + ", "
        + str(num_cuts)
        + ", "
        + str(gap_closed_all)
        + ", "
        + str(separator.model.MIPGap)
        + "\n"
    )

    with open(outputfile, "a") as f:
        f.write(line)

    new_solution = np.array([var.X for var in lp.getVars()])
    comparisons = [
        np.allclose(old_solution[i], new_solution, atol=1.0e-4)
        for i in range(len(new_solution))
    ]

    # Check termination criteria
    if rounds >= max_rounds:
        continuar = False
        print("reached maximum number of rounds ", max_rounds)
        with open(problemfile, "a") as f:
            f.write("Reached maximum number of rounds " + str(max_rounds) + "\n")

    if gap_closed_all >= 100:
        continuar = False
        print("closed all gap")
        with open(problemfile, "a") as f:
            f.write("closed all gap\n")

    if np.allclose(new_solution, solution):
        continuar = False
        print("point is not separated")
        with open(problemfile, "a") as f:
            f.write("Point is not separated\n")

    # Else, continue
    if continuar:
        solution = copy.deepcopy(new_solution)
        rounds = rounds + 1
        tic = time.time()
        nic = time.process_time()
        separator.update_solution(solution)

        this_dataset, feature_names = features(
            lp, solution, var_types, orig=ip.NumConstrs
        )
        this_dataset = pd.DataFrame(this_dataset, columns=feature_names)
        this_dataset["instance_id"] = [instance_idx] * this_dataset.shape[0]
        this_dataset["cut_iter"] = [rounds] * this_dataset.shape[0]

        scalar = StandardScaler()
        this_dataset = scaler.fit_transform(this_dataset)

        predictions = learned_model.predict_proba(this_dataset)
        predictions = [1 if p[1] > opt_threshold else 0 for p in predictions]

        toc = time.time()
        noc = time.process_time()
        print("updated solution in ", toc - tic, "wall seconds")
        print("updated solution in ", noc - nic, "cpu seconds")
