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

import argparse
from utils import io, plot

parser = argparse.ArgumentParser()

parser.add_argument('--path_chip_db', required=True, type=str, help='Path to chip database')
parser.add_argument('--path_clustering_output', required=True, type=str, help='Path to clustering output file')
parser.add_argument('--modality', required=True, type=str, help='Clustering modality to plot, corresponding to a column in the clustering output file')

parser.add_argument('--figure_name', default='overlay.png', type=str, help='File name to save overlay image')
parser.add_argument('--min_cluster_members', default=3, type=int, help='Minimum number of chips in a cluster required to appear in the plot')
parser.add_argument('--cmap', default='tab10', type=str, help='Color map used for plotting')
parser.add_argument('--dpi', default=150, type=int, help='dpi of output image')
parser.add_argument('--W', default=6., type=float, help='Width of output image')
parser.add_argument('--H', default=6., type=float, help='Height of output image')

args = parser.parse_args()

print("Loading clustering results")
print(f"...chip database: {args.path_chip_db}")
print(f"...clustering output: {args.path_clustering_output}")
results = io.read_results(args.path_chip_db, args.path_clustering_output)

print(f"Plotting overlay of clustering using {args.modality} modality")
plot.overlay(results, args)

print(f"Overlay plot saved to {args.figure_name}.")
