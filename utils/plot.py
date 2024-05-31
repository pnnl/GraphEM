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
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from skimage import transform

def format_chips(results, shape, dim):
    chips = np.zeros(np.append(shape, [3]), dtype='int')
    
    for i, drow in results.sort_values(['row','col']).iterrows():
        r = drow['row']
        rstart = r+((dim-1)*r)
        rend = r+((dim-1)*r)+dim
        
        c = drow['col']
        cstart = c+((dim-1)*c)
        cend = c+((dim-1)*c)+dim
    
        img = np.array(drow['image data']).reshape(dim, dim, 3)
        chips[rstart:rend,cstart:cend,:] = img
    return chips

def overlay(results, args):
    # enumerate chips in each row and column
    n_rows = len([x for x in results['chip ID'].tolist() if 'R0C' in x])
    n_columns = len([x for x in results['chip ID'].tolist() if x[-2:]=='C0'])

    # create grid to upscale cluster blocks to match size of image
    grid = np.zeros((results['col'].max()+1,results['row'].max()+1), dtype='int')
    for c in results[args.modality].value_counts().index:
        grid[results.loc[results[args.modality]==c]['row'].to_numpy(),results.loc[results[args.modality]==c]['col'].to_numpy()]=c

    # filter clusters with too few members
    vc = results[args.modality].value_counts()
    max_cluster_idx = len(vc.loc[vc>=args.min_cluster_members])+1

    # plot definition
    fig = plt.figure(figsize=(args.W, args.H), dpi=args.dpi)
    ax = fig.gca()
    
    # resize to fit image
    dim = int(np.sqrt(np.array(results.iloc[0]['image data']).reshape(-1,3).shape[0]))
    updim = (int(n_rows*dim), int(n_columns*dim))
    g = transform.resize_local_mean(grid, updim, preserve_range=True).astype('int')

    # apply mask
    g=np.ma.masked_where(g>max_cluster_idx, g)

    # plot chips
    chips = format_chips(results, shape=np.array(updim), dim=dim)
    ax.imshow(chips, interpolation='none', cmap=cm.gray)

    # plot cluster blocks
    ax.imshow(g, alpha=0.5, cmap=args.cmap, interpolation='none', 
              norm = matplotlib.colors.BoundaryNorm(range(0,max_cluster_idx+1), ncolors=max_cluster_idx)
              )

    # plot and save
    plt.axis('off')
    plt.margins(0.1)
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(args.figure_name, pad_inches=0, dpi=args.dpi)
    plt.show()