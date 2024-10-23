# import sys
# !{sys.executable} -m pip install 
import pandas as pd
from tqdm import tqdm
import numpy as np
from geopy.distance import geodesic
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: default, 1: no INFO, 2: no INFO or WARNING, 3: no INFO, WARNING, or ERROR
import warnings
warnings.filterwarnings("ignore", message="oneDNN custom operations are on.*")
warnings.filterwarnings("ignore", message="Could not find cuda drivers.*")
warnings.filterwarnings("ignore", message="This TensorFlow binary is optimized.*")
warnings.filterwarnings("ignore", message="TF-TRT Warning: Could not find TensorRT.*")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
np.random.seed(42)








'''
Load RGI -- loops through the unzipped CSV attribute files of RGI glaciers and loads them into a pandas dataframe.
'''
def load_RGI(RGI_path):
    df = pd.DataFrame()
    for RGI_file in os.listdir(RGI_path):
        if RGI_file.endswith('.csv'):
            dft = pd.read_csv(
                os.path.join(RGI_path,RGI_file),
                encoding_errors = 'replace', 
                on_bad_lines = 'skip'
            )
            df = pd.concat([df,dft],axis = 0)
    df = df.sort_values('RGIId').reset_index()
    # df['region'] = df['RGIId'].str[6:8]
    df = pd.concat([df,pd.Series(df['RGIId'].str[6:8],name = 'region')],axis = 1)
    df = df.drop('index',axis = 1)
    return df


'''
GlaThiDa_RGI_index_matcher: uses multiprocessing or not. Go through GlaThiDa,
and search for the nearest RGI glacier. Measure the distance and append.
'''
def match_GlaThiDa_RGI_index(
#     pth = '/data/fast1/glacierml/data/',
    RGI,glathida,
    verbose = False,
    useMP = False
):
    # 

    
    if useMP == False:
        centroid_distances = []
        RGI_ids = []
        for i in tqdm(glathida.index):
            RGI_id_match, centroid_distance = get_id(RGI,glathida,verbose,i)
            centroid_distances.append(centroid_distance)
            RGI_ids.append(RGI_id_match)
        df = pd.concat(
            [
                glathida,
                pd.Series(RGI_ids,name = 'RGIId'),
                pd.Series(centroid_distances,name = 'centroid distance')
            ],axis = 1
        )
        
            
        return df
            
    else:
        from functools import partial
        import multiprocessing
        pool = multiprocessing.Pool(processes=4)         # create a process pool with 4 workers
        newfunc = partial(get_id,RGI,glathida,verbose) #now we can call newfunc(i)
        output = pool.map(newfunc, glathida.index)
        for i in tqdm(glathida.index):      

            glathida.loc[glathida.index[i], 'RGIId'] = output[i][0]
            glathida.loc[glathida.index[i], 'RGI Centroid Distance'] = output[i][1]
        return glathida
    
    
''' 
Works with match function above.
'''
def get_id(RGI,glathida,verbose,i):
    if verbose: print(f'Working on Glathida index {i}')
    #obtain lat and lon from glathida 
    glathida_ll = (glathida.loc[i].LAT,glathida.loc[i].LON)

    # find distance between selected glathida glacier and all RGI
    distances = RGI.apply(
        lambda row: geodesic((row.CenLat,row.CenLon),glathida_ll),
            axis = 1
    )
    centroid_distance = distances.min()
    min_indices = np.where(distances == centroid_distance)

    RGI_index = pd.Series(min_indices, name = 'RGI_indexes')
    number_glaciers_matched = len(RGI_index)
    print(number_glaciers_matched)
    if len(RGI_index) == 1:
        RGI_id_match = (RGI['RGIId'].iloc[RGI_index.loc[0]])
    else:
        RGI_id_match = -1
        centroid_distance = -1
        
    return RGI_id_match, centroid_distance





'''
Loads matched pickle file, merges with RGI attributes and keeps relevant information.
'''
def load_training_data(data_path):  
    # Read matched dataset and select data
    df = pd.read_pickle(os.path.join(data_path,'matched.pkl'))
    df = df[[
        'RGIId','RGI Centroid Distance','SURVEY_DATE',
        'AREA','MEAN_THICKNESS_UNCERTAINTY','GLACIER_NAME','MEAN_THICKNESS'
    ]]
    df = df.rename(columns = {
        'MEAN_THICKNESS':'Thickness',
    })
    
    # Load RGI to get RGI attributes
    RGI_path = os.path.join(data_path,'RGI')
    RGI = load_RGI(RGI_path)

    # Put the data together
    df = pd.merge(df,RGI,how = 'inner', on = 'RGIId')
    
    return df

'''
apply uncertainty thresholds to training data
'''
def coregister_data(
    data_path,
    coregistration = '1',
):
    
    df = load_training_data(data_path)
    
    
    import configparser
    config = configparser.ConfigParser()
    config.read('model_coregistration.txt')   
    
    
    # df['RGI Centroid Distance'] = df['RGI Centroid Distance'].astype(str).str[:-2].astype(float)
    df['RGI Centroid Distance'] = df['RGI Centroid Distance'] * 1e3

    rad = np.sqrt((df['Area']*1e6) / np.pi)
    radius = pd.Series(rad,name = 'AVG Radius').astype(float)
    
    df = pd.concat([df,radius],axis = 1)
    distance_test = pd.Series(
        df['RGI Centroid Distance'] / df['AVG Radius'],
        name = 'distance test'
    )
    
    
    df = pd.concat([df,distance_test],axis = 1)
    
    df = df.drop(
        df[df['distance test'] >= float(config[coregistration]['distance threshold'])].index
    )
    
    # Look at size difference between data
    ps = pd.Series(
        (df['Area'] - df['AREA']) / df['AREA'],
        name = 'perc difference'
    )
    df = pd.concat([df,ps],axis = 1)
    area_scrubber = config[coregistration]['area scrubber']
    if area_scrubber == 'on':
        size_threshold = float(config[coregistration]['size threshold'])
        df = df[
            (df['perc difference'] <= size_threshold) & 
            (df['perc difference'] >= -size_threshold)
        ]
    
    df = df.drop(df[df['RGIId'].duplicated(keep = False)].index)

    df = df[[
        'RGIId','CenLat','CenLon','Slope','Zmin','Zmed','Zmax',
        'Area','Aspect','Lmax','Thickness'
    ]]
    df = df.sort_values('RGIId')
    df = df.reset_index().drop('index',axis = 1)
    return df


        
'''

input = name of dataframe and selected random state.
output = dataframe and series randomly selected and populated as either training or test features or labels
'''
# Randomly selects data from a df for a given random state (usually iterated over a range of 25)
# Necessary variables for training and predictions
def split_data(df, random_state = 0,var = 'Thickness'):
    train_dataset = df.sample(frac=0.7, random_state=random_state).drop('RGIId', axis = 1)
    test_dataset = df.drop(train_dataset.index).drop('RGIId', axis = 1)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    #define label - attribute training to be picked
    train_labels = train_features.pop(var)
    test_labels = test_features.pop(var)
    
    return train_features, test_features, train_labels, test_labels








'''
build_linear_model
input = normalized data and desired learning rate
output = linear regression model
'''
# No longer used
def build_linear_model(normalizer,learning_rate=0.01):
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(1)
    ])

    model.compile( 
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_absolute_error')
    
    return model



'''
build_dnn_model
input = normalized data and selected learning rate
output = dnn model with desired layer architecture, ready to be trained.
'''
def build_dnn_model(
    norm, learning_rate=0.01, layer_1 = 10, layer_2 = 5, loss = 'mae'
):    
    model = keras.Sequential(
        [
              norm,
              layers.Dense(layer_1, activation='relu'),
              layers.Dropout(rate = 0.1, seed = 0),
              layers.Dense(layer_2, activation='relu'),
              layers.Dense(1) 
        ]
    )    
    if loss == 'mse':
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate)
        )   
    if loss == 'mae':
        
        model.compile(
            loss='mean_absolute_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate)
        )      
    return model



'''
plot_loss
input = desired test results
output = loss plots for desired model
'''
def plot_loss(history):
    plt.plot(
        history['loss'], 
         label='loss',
        color = 'blue'
    )
    plt.plot(
        history['val_loss'], 
        label='val_loss',
        color = 'orange'
    )
    #   plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid(True)
    plt.title('Model Training')
#     plt.show()
    

def load_dnn_model(
    model_loc
):
    
    dnn_model = tf.keras.models.load_model(model_loc)
    
    return dnn_model
    
    

def findlog(x):
    if math.isnan(x):
        log = 0
    elif x > 0:
        log = math.log(x)
    elif x < 0:
        log = math.log(-x) * -1
    else:  # x == 0
        log = 0
    return log



def load_LOO_data(
    home_path,
    include_train = False,
    include_refs = False
):

    df = pd.read_pickle(os.path.join(home_path,'models','LOO','rgi_est_raw.pkl')) 
    
    cols = []
    for i in range(202):
        cols.append(i)
    df[cols] = np.round(df[cols],0)
    df[cols] = df[cols] / 1e3
    
    
    if include_refs == True:
        ref = pd.read_pickle(os.path.join(home_path,'data/ref/refs.pkl'))
        df = pd.merge(df, ref, how = 'left', on = 'RGIId')
        df['FMT'] = np.round(df['FMT'],0)
        df['FMT'] = df['FMT'] / 1e3   
        
    if include_train == True:
        train = coregister_data(home_path + '/data/', '4')
        train = train.sample(frac = 1,random_state = 0)
        train = train.reset_index().drop('index', axis = 1)
        df = pd.merge(train, df, how = 'inner', on = list(train)[:-1])
        df['Thickness'] = df['Thickness'] / 1e3
        
    
    
        
    return cols, df








