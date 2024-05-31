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
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
from skimage import io


def img_embedding(image, preprocess_input, base_model, model_input_size=(32,32)):
    x = tf.image.resize(image, size=model_input_size)
    x = preprocess_input(x)
    x = base_model(np.expand_dims(x, axis=0))
    return x 

def save_chip_database(database_path, df):
    # save chips as h5 database
    hf = h5py.File(database_path, 'a')  
    cols = df.columns[1:]
    
    for i, row in df.iterrows(): 
        # create chip name
        name = row['chip ID']
        subgrp = hf.create_group(name) 

        # save chip data
        for col in cols:
            _ = subgrp.create_dataset(col, data=row[col])

    hf.close()
    
def collect_data(chip_dir, eds_data, atom_data, database):
    # load chipped images 
    x = glob.glob(os.path.join(chip_dir,'*.jpg'))
    chip_dict = {i.split('/')[-1].replace('.jpg',''):io.imread(i) for i in x}
    
    # create dataframe with data
    chip_df = pd.DataFrame()
    chip_df.index = chip_dict.keys()
    chip_df['chip ID'] = chip_dict.keys()
    chip_df['image data'] = chip_dict.values()
    chip_df['image data'] = chip_df['image data'].apply(lambda x: x.flatten())
    
    # extract image embeddings using pretrained VGG16
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications
    model_input_shape = (32,32,3)
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    base_model = tf.keras.applications.vgg16.VGG16(input_shape=model_input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    # add embeddings to dataframe
    chip_emb_dict={}
    for k,v in chip_dict.items():
        chip_emb_dict[k] = img_embedding(v, preprocess_input, base_model, model_input_shape[:2]).numpy().flatten()
    
    chip_df['image encoding data'] = chip_emb_dict.values()
    chip_df['image encoding data'] = chip_df['image encoding data'].apply(lambda x: x.flatten())
    
    # add eds spectra to dataframe
    d = np.load(eds_data)
    edsspect_dict = dict(map(lambda i,j : (i,j) , d['chips'] ,d['spectrum']))
    chip_df['eds spectra'] = chip_df.apply(lambda row: edsspect_dict[row['chip ID']], axis=1)
    
    # add atom percent data to dataframe
    tmp = pd.read_csv(atom_data)
    cols = tmp.columns[2:]
    atmp = {row['Chip_ID']:row[cols].to_numpy().astype(float) for i,row in tmp.iterrows()}
    chip_df['atom percent data'] = chip_df.apply(lambda row: atmp[row['chip ID']], axis=1)
    
    save_chip_database(database, chip_df)