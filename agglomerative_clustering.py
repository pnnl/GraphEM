"""
                             DISCLAIMER: 
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
"""

import numpy as np
import pandas as pd
from sklearn import cluster
import argparse
from utils.io import read_chip_database

parser = argparse.ArgumentParser()

parser.add_argument('--path_chip_db', required=True, type=str, help='Path to chip database')
parser.add_argument('--path_clustering_output', required=True, type=str, help='Path to save clustering output file')
parser.add_argument('--c', required=True, type=int, help='Number of expected clusters')

args = parser.parse_args()

print(f"Loading chip database from {args.path_chip_db}")
chip_df = read_chip_database(args.path_chip_db)

print(f"Clustering individual modalities using n_clusters={args.c}")
results = pd.DataFrame()
results["chip ID"] = chip_df["chip ID"]

labels = ['haadf', 'atom', 'eds']
for i,f in enumerate(["image encoding data", "atom percent data", "eds spectra"]):
    chip_embeddings = np.vstack(chip_df[f].to_numpy())
    labels = cluster.AgglomerativeClustering(n_clusters=args.c).fit(chip_embeddings)
    results[labels[i]] = labels.labels_

print(f"Ensembling modality pairs")
ensemble_pairs = [("haadf", "atom"),("haadf", "eds"),("atom", "eds")]
for pair in ensemble_pairs:
    a = pair[0]
    b = pair[1]

    col=a+'+'+b
    results[col] = results.apply(lambda row: (row[a],row[b]), axis=1)
    results[col] = results[col].astype('category').cat.codes

results.to_csv(args.path_clustering_output, index=False)
print(f"Clustering output saved to {args.path_clustering_output}.")
