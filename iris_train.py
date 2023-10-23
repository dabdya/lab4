from sklearn.linear_model import LogisticRegression
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import pickle


ERR_TRAIN_DATA_NOT_FOUND = 1


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--train_path", type = Path, required = True)
    argparser.add_argument("--save_dir", type = Path, required = True)
    return argparser.parse_args()


def try_to_load_train_data(train_path: Path):
    if not train_path.exists():
        import sys
        print(f"Failed to load train data from {train_path}")
        sys.exit(ERR_TRAIN_DATA_NOT_FOUND)
    
    return pd.read_csv(train_path)


def train(df: pd.DataFrame):
    reg = LogisticRegression()
    return reg.fit(df.drop(["target"], axis = 1).values, df.target.values)


def save(model, path: Path):
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path/"model.pickle", "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    args = parse_args()

    df = try_to_load_train_data(args.train_path)
    model = train(df)
    save(model, args.save_dir)
    