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

import pandas as pd
import h5py

def read_chip_database(database_path):
    # convert h5 database into dataframe
    with h5py.File(database_path, 'r') as f:
        chips = list(f.keys())
        df = pd.DataFrame({'chip ID': chips})

        for col in f[chips[0]].keys():
            values=[]
            for i, row in df.iterrows():
                values.append(f[row['chip ID']][col][:])
            df[col] = values
            
    return df


def read_results(path_chip_db, path_clustering_output):
    # load clustering results and chips
    chip_df = read_chip_database(path_chip_db)
    results = pd.read_csv(path_clustering_output)
    
    results['row'] = results['chip ID'].apply(lambda x: int(x.split('C')[0].replace('R','')))
    results['col'] = results['chip ID'].apply(lambda x: int(x.split('C')[1]))
    
    results = results.merge(chip_df[['chip ID','image data']], on='chip ID')
    return results