import mlflow
import mlflow.sklearn
from pathlib import Path
from shutil import copy2, rmtree
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


mlflow.set_tracking_uri("http://127.0.0.1:5000")
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
STAGING_DIR = PROJECT_DIR / "artifact_staging"


def stage_project_artifacts() -> Path:
    if STAGING_DIR.exists():
        rmtree(STAGING_DIR)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    allowed_extensions = {".py", ".png", ".html", ".yaml", ".yml"}
    excluded_names = {"mlruns", "mlartifacts", "artifact_staging"}

    for path in PROJECT_DIR.rglob("*"):
        if any(part in excluded_names for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in allowed_extensions:
            relative_path = path.relative_to(PROJECT_DIR)
            destination = STAGING_DIR / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(path, destination)

    return STAGING_DIR


# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 10

# Mention your experiment below
mlflow.autolog
mlflow.set_experiment('yt-mlflow')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    mlflow.log_params({
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "test_size": 0.10,
        "random_state": 42,
    })
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(rf, name="model")

    # creating confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.title('confusion matrix')
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    mlflow.log_artifact(confusion_matrix_path)
    mlflow.log_artifacts(str(stage_project_artifacts()), artifact_path="project_files")
    mlflow.set_tags({"author": "vikash", "project": "wine classification"})
    mlflow.sklearn.log_model(rf, name="random_forest")
print(accuracy)
import mlflow
import mlflow.sklearn
from pathlib import Path
from shutil import copy2, rmtree
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


mlflow.set_tracking_uri("http://127.0.0.1:5000")
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
STAGING_DIR = PROJECT_DIR / "artifact_staging"


def stage_project_artifacts() -> Path:
    if STAGING_DIR.exists():
        rmtree(STAGING_DIR)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    allowed_extensions = {".py", ".png", ".html", ".yaml", ".yml"}
    excluded_names = {"mlruns", "mlartifacts", "artifact_staging"}

    for path in PROJECT_DIR.rglob("*"):
        if any(part in excluded_names for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in allowed_extensions:
            relative_path = path.relative_to(PROJECT_DIR)
            destination = STAGING_DIR / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(path, destination)

    return STAGING_DIR

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 10

# Mention your experiment below
mlflow.autolog
mlflow.set_experiment('yt-mlflow')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    mlflow.log_params({
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "test_size": 0.10,
        "random_state": 42,
    })
    mlflow.log_metric('accuracy',accuracy)
    mlflow.sklearn.log_model(rf, "model")

    # creating confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.title('confusion matrix')
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    mlflow.log_artifact(confusion_matrix_path)
    mlflow.log_artifacts(str(stage_project_artifacts()), artifact_path="project_files")
    mlflow.set_tags({"author": "vikash", "project": "wine classification"})
    mlflow.sklearn.log_model(rf,"random forest model")
print(accuracy)