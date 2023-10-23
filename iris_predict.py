from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import pickle


ERR_MODEL_NOT_FOUND = 1
ERR_TEST_DATA_NOT_FOUND = 2


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--test_path", type = Path, required = True)
    argparser.add_argument("--model_path", type = Path, required = True)
    argparser.add_argument("--save_dir", type = Path, required = True)
    return argparser.parse_args()


def try_to_load_model(model_path: Path):
    if not model_path.exists():
        import sys
        print(f"Failed to load model from {model_path}")
        sys.exit(ERR_MODEL_NOT_FOUND)

    with open(model_path, "rb") as file:
        return pickle.load(file)
    

def try_to_load_test_data(test_path: Path):
    if not test_path.exists():
        import sys
        print(f"Failed to load test data from {test_path}")
        sys.exit(ERR_TEST_DATA_NOT_FOUND)
    return pd.read_csv(test_path)


def save(predictions, path: Path):
    pd.DataFrame({"predictions": predictions}).to_csv(path/"predict.csv")


if __name__ == "__main__":
    args = parse_args()

    model = try_to_load_model(args.model_path)
    df = try_to_load_test_data(args.test_path)
    predictions = model.predict(df.drop(["target"], axis = 1).values)
    save(predictions, args.save_dir)
