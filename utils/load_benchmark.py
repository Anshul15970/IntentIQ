import pandas as pd


def load_benchmark():

    df = pd.read_csv("results/benchmark_results.csv")

    return df