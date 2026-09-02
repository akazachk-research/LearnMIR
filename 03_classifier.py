import os
import argparse
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import TunedThresholdClassifierCV

parser = argparse.ArgumentParser()
parser.add_argument("-problem", type=int, default=2)
args = parser.parse_args()

base_dir = "./goodinstances/"
all_instances = os.listdir(base_dir)
for file in all_instances:
    if ".csv" in file:
        all_instances.remove(file)
all_instances.sort()

problem_name = all_instances[args.problem].split(sep=".")[0]

directory = (
    # "/blue/akazachkov/o.guaje/" + "gooddata/" + problem_name + "/"
    "./goodfiles/"
    # "/home/oguaje/scratch/goodfiles/"
    + problem_name
    + "/"
)

data_dir = directory + "data/"

files_to_read = os.listdir(data_dir)

all_datasets = []

# files_to_read = files_to_read[:5]

for file in files_to_read:
    this_data = pd.read_csv(data_dir + file)
    all_datasets.append(this_data)


my_data = pd.concat(all_datasets, ignore_index=True)
scaler = StandardScaler()


instances = my_data["instance_id"].unique()
split = int(0.8 * len(instances))
train_instances = instances[:split]
test_instances = instances[split:]

y = my_data["label_any"]
class1_percent = sum(y) / len(y)
class0_count = 1 - class1_percent


with open(directory + "train.txt", "w") as f:
    f.write(str(list(train_instances)))

with open(directory + "test.txt", "w") as f:
    f.write(str(list(test_instances)))

train_data = my_data[my_data["instance_id"].isin(train_instances)]
test_data = my_data[my_data["instance_id"].isin(test_instances)]

y_train = train_data["label_any"]
X_train = train_data.drop(
    columns=[
        "variable_id",
        "instance_id",
        "label_any",
        "label_eps",
        "label_max",
    ]
)
y_test = test_data["label_any"]
X_test = test_data.drop(
    columns=[
        "variable_id",
        "instance_id",
        "label_any",
        "label_eps",
        "label_max",
    ]
)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
sample_weights = compute_sample_weight("balanced", y_train)

clf = GradientBoostingClassifier(max_depth=5, random_state=1603).fit(
    X_train, y_train, sample_weight=sample_weights
)

train_f1 = metrics.f1_score(y_train, clf.predict(X_train))
train_acc = metrics.accuracy_score(y_train, clf.predict(X_train))
train_precision = metrics.precision_score(y_train, clf.predict(X_train))
train_recall = metrics.recall_score(y_train, clf.predict(X_train))
test_acc = metrics.accuracy_score(y_test, clf.predict(X_test))
test_f1 = metrics.f1_score(y_test, clf.predict(X_test))
test_precision = metrics.precision_score(y_test, clf.predict(X_test))
test_recall = metrics.recall_score(y_test, clf.predict(X_test))

train_conf_matrix = metrics.confusion_matrix(y_train, clf.predict(X_train))
test_conf_matrix = metrics.confusion_matrix(y_test, clf.predict(X_test))

tuned_clf = TunedThresholdClassifierCV(clf, scoring="f1").fit(X_train, y_train)

tuned_train_f1 = metrics.f1_score(y_train, clf.predict(X_train))
tuned_train_acc = metrics.accuracy_score(y_train, clf.predict(X_train))
tuned_train_precision = metrics.precision_score(y_train, clf.predict(X_train))
tuned_train_recall = metrics.recall_score(y_train, clf.predict(X_train))
tuned_test_acc = metrics.accuracy_score(y_test, clf.predict(X_test))
tuned_test_f1 = metrics.f1_score(y_test, clf.predict(X_test))
tuned_test_precision = metrics.precision_score(y_test, clf.predict(X_test))
tuned_test_recall = metrics.recall_score(y_test, clf.predict(X_test))

tuned_train_conf_matrix = metrics.confusion_matrix(
    y_train, clf.predict(X_train)
)
tuned_test_conf_matrix = metrics.confusion_matrix(y_test, clf.predict(X_test))
optimal_threshold = tuned_clf.best_threshold_
print(optimal_threshold)

output_file = directory + "learning_metrics.txt"
output_model = directory + "classifier.joblib"
otuput_threshold_file = directory + "optimal_threshold.txt"

with open(otuput_threshold_file, "w") as f:
    f.write(str(optimal_threshold))

with open(output_model, "wb") as f:
    dump(tuned_clf, f)

with open(output_file, "w") as f:
    f.write("Metrics before tuning:\n")
    f.write(f"Train F1 Score: {train_f1}\n")
    f.write(f"Train Accuracy: {train_acc}\n")
    f.write(f"Train Precision: {train_precision}\n")
    f.write(f"Train Recall: {train_recall}\n")
    f.write(f"Test Accuracy: {test_acc}\n")
    f.write(f"Test F1 Score: {test_f1}\n")
    f.write(f"Test Precision: {test_precision}\n")
    f.write(f"Test Recall: {test_recall}\n")
    f.write("Train Confusion Matrix:\n")
    f.write(str(train_conf_matrix) + "\n")
    f.write("Test Confusion Matrix:\n")
    f.write(str(test_conf_matrix) + "\n")
    f.write("\nMetrics after tuning:\n")
    f.write(f"Tuned Train F1 Score: {tuned_train_f1}\n")
    f.write(f"Tuned Train Accuracy: {tuned_train_acc}\n")
    f.write(f"Tuned Train Precision: {tuned_train_precision}\n")
    f.write(f"Tuned Train Recall: {tuned_train_recall}\n")
    f.write(f"Tuned Test Accuracy: {tuned_test_acc}\n")
    f.write(f"Tuned Test F1 Score: {tuned_test_f1}\n")
    f.write(f"Tuned Test Precision: {tuned_test_precision}\n")
    f.write(f"Tuned Test Recall: {tuned_test_recall}\n")
    f.write("Tuned Train Confusion Matrix:\n")
    f.write(str(tuned_train_conf_matrix) + "\n")
    f.write("Tuned Test Confusion Matrix:\n")
    f.write(str(tuned_test_conf_matrix) + "\n")
    f.write(f"Optimal Threshold: {optimal_threshold}\n")
