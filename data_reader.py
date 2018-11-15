import pandas as pd


def read(file_names, labels):
    df = pd.DataFrame()
    for label, file_name in zip(labels, file_names):
        file = open(file_name, 'r')
        lines = file.readlines()
        for line in lines:
            df = df.append([[line, label]], ignore_index=True)
        file.close()
    return df
