from sklearn import datasets
from sklearn import model_selection
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--random_state", type = int, required = False, default = 42)
    argparser.add_argument("--test_size", type = float, required = False, default = 0.25)
    argparser.add_argument("--save_dir", type = Path, required = True)
    return argparser.parse_args()


def train_test_split(dataset, test_size, random_state):
    return model_selection.train_test_split(
        dataset.data, dataset.target, 
        test_size = test_size, random_state = random_state, shuffle = True, stratify = dataset.target)


def save(X, y, path: Path):
    path.parent.mkdir(parents = True, exist_ok = True)
    df = pd.DataFrame(
        data = {feature: X[:, i] for i, feature in enumerate(iris.feature_names)})
    df["target"] = y
    df.to_csv(path)


if __name__ == "__main__":

    iris = datasets.load_iris()
    args = parse_args()

    X_train, X_test, y_train, y_test = train_test_split(
        dataset = iris, test_size = args.test_size, random_state = args.random_state
    )

    save(X_train, y_train, args.save_dir/"train.csv")
    save(X_test, y_test, args.save_dir/"test.csv")
