from pathlib import Path
from typing import Union
from matplotlib.figure import Figure


class Report:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir

    def add(self, object: Union[Figure, str], filename: str):
        if isinstance(object, Figure):
            object.savefig(self.save_dir/filename)
        else:
            with open(self.save_dir/filename, "a") as report:
                report.write(object)
                report.write("\n")
