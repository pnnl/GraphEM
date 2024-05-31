import h5py
import numpy as np
import glob
from skimage import io
import tensorflow as tf
import os
import pandas as pd


def save_chip_database(database_path, df):
    hf = h5py.File(database_path, 'a')  
    
    cols = df.columns[1:]
    
    for i, row in df.iterrows(): 
        
        ### Create chip name
        name = row['chip ID']
        subgrp = hf.create_group(name) 

        ### Save chip data
        for col in cols:
            _ = subgrp.create_dataset(col, data=row[col])

    hf.close()

def read_chip_database(database_path):
    with h5py.File(database_path, 'r') as f:
        chips = list(f.keys())
        df = pd.DataFrame({'chip ID': chips})

        for col in f[chips[0]].keys():
            values=[]
            for i, row in df.iterrows():
                values.append(f[row['chip ID']][col][:])
            df[col] = values
            
    return df



def switch_rows_and_cols(label):
        i=label.split('C')
        return f"R{i[1]}C{i[0][1:]}"
    
def collect_data(base_dir, tag, eds_tag):
    # chip embeddings
    x=glob.glob(os.path.join(base_dir,f'chips/{tag}*/*.jpg'))
    chip_dict={i.split('/')[-1].replace('.jpg',''):io.imread(i) for i in x}
    
    # create dataframe with data
    chip_df=pd.DataFrame()
    chip_df.index=chip_dict.keys()
    chip_df['chip ID']=chip_dict.keys()
    chip_df['image data']=chip_dict.values()
    chip_df['image data']=chip_df['image data'].apply(lambda x: x.flatten())
    
    #### image embeddings -- mobilenetv2 ####
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    base_model = tf.keras.applications.vgg16.VGG16(input_shape=(32,32,3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    def img_embedding(image, base_model, preprocess_input):
        x=tf.image.resize(image,size=(32,32))
        x=preprocess_input(x)
        x=base_model(np.expand_dims(x, axis=0))
        return x 
    
    chip_emb_dict={}
    for k,v in chip_dict.items():
        chip_emb_dict[k]=img_embedding(v, base_model, preprocess_input).numpy().flatten()
    
    chip_df['image encoding data']=chip_emb_dict.values()
    chip_df['image encoding data']=chip_df['image encoding data'].apply(lambda x: x.flatten())
    
    #### eds spectra ####
    x=glob.glob(os.path.join(base_dir,f'eds/*{eds_tag}/*.npz'))
    d = np.load(x[0])

    edsspect_dict = dict(map(lambda i,j : (i,j) , [switch_rows_and_cols(i) for i in d['chips']] ,d['spectrum']))
    chip_df['eds spectra']=chip_df.apply(lambda row: edsspect_dict[row['chip ID']], axis=1)
    
    #### spec2vec ####
    #x=glob.glob(os.path.join(base_dir,f'spec2vec/{tag}/*.npz'))
    #d = np.load(x[0])
    #spect_dict = dict(map(lambda i,j : (i,j) , [switch_rows_and_cols(i) for i in d['Chip_ID']], d['Embedding']))
    #chip_df['eds encoding data']=chip_df.apply(lambda row: spect_dict[row['chip ID']], axis=1)
    
    ### 'atom percent data' ###
    x=glob.glob(os.path.join(base_dir,f'eds/*{eds_tag}/Atomic*.csv'))
    tmp = pd.read_csv(x[0])
    cols = tmp.columns[2:]
    atmp = {row['Chip_ID']:row[cols].to_numpy().astype(float) for i,row in tmp.iterrows()}
    chip_df['atom percent data']=chip_df.apply(lambda row: atmp[row['chip ID']], axis=1)
    
    save_chip_database(os.path.join(base_dir, 'hdf5s', f'{tag}-vgg16.h5'), chip_df)