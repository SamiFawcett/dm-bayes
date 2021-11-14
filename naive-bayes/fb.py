import numpy as np
import pandas as pd


def parse(filename):
    headers = list(pd.read_csv(filename, nrows=0).columns)
    headers.pop(len(headers) - 1)
    headers.pop(0)
    data = pd.read_csv(filename, usecols=headers).to_numpy()
    unscaled_column_view = np.transpose(data)

    return [unscaled_column_view, data, headers]


if __name__ == "__main__":

 filename('../energydata_')
