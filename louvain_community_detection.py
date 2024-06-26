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

import os
import louvain
import pandas as pd
import numpy as np
import igraph as ig
from scipy.spatial.distance import pdist, squareform
from data_tools import read_chip_database

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path_chip_db', required=True, type=str, help='Path to chip database')
parser.add_argument('--path_clustering_output', required=True, type=str, help='Path to save clustering output file')
parser.add_argument('--c', required=True, type=float, help='Cutoff between 0 and 1')

args = parser.parse_args()

print(f"Loading chip database from {args.path_chip_db}")
chip_df = read_chip_database(args.path_chip_db)

m = sum(chip_df["chip ID"].str.count("R\d+C0"))
n = sum(chip_df["chip ID"].str.count("R0C\d+"))
print("Number of rows: ", m)
print("Number of columns: ", n)
d = {}

_ = 0
for i in range(m):
    for j in range(n):
        d["R" + str(i) + "C" + str(j)] = _
        _ += 1

inv_d = {v: k for k, v in d.items()}

chip_df = chip_df.replace({"chip ID": d}).sort_values(by=['chip ID']).replace({"chip ID": inv_d})

print(f"Clustering individual modalities using cutoff={args.c}")
results = pd.DataFrame()
results["chip ID"] = chip_df["chip ID"]

sims = {}
labels = ['haadf', 'atom', 'eds']
for i,f in enumerate(["image encoding data", "atom percent data", "eds spectra"]):
    sim = squareform(1 - pdist(np.stack(chip_df[f].values), metric="cosine"))
    adj = (sim >= args.c).astype(int)
    g = ig.Graph.Adjacency(adj, mode='undriected')
    part = louvain.find_partition(g, louvain.ModularityVertexPartition, seed=42)
    pred = {}
    for s in range(len(part)):
        for node in part[s]:
            pred[node] = s
    sims[labels[i]] = sim
    df = pd.DataFrame({"chip ID": pred.keys(), labels[i]: pred.values()})
    df = df.sort_values(by=['chip ID']).replace({"chip ID": inv_d})
    results = results.merge(df, left_on="chip ID", right_on="chip ID")

print(f"Ensembling modality pairs")
ensemble_pairs = [("haadf", "atom"),("haadf", "eds"),("atom", "eds")]
for pair in ensemble_pairs:
    a = pair[0]
    b = pair[1]
    
    avg_sim = (sims[a] + sims[b]) / 2
    adj = (sim >= args.c).astype(int)
    g = ig.Graph.Adjacency(adj, mode='undriected')
    part = louvain.find_partition(g, louvain.ModularityVertexPartition, seed=42)
    pred = {}
    for s in range(len(part)):
        for node in part[s]:
            pred[node] = s
    sims[labels[i]] = sim
    col=a+'+'+b
    df = pd.DataFrame({"chip ID": pred.keys(), col: pred.values()})
    df = df.sort_values(by=['chip ID']).replace({"chip ID": inv_d})
    results = results.merge(df, left_on="chip ID", right_on="chip ID")
results.to_csv(os.path.join(args.path_clustering_output, 'results.csv'), index=False)
print(f"Clustering output saved to {args.path_clustering_output}.")