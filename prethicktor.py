import glacierml as gl
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.util import deprecation
import logging
import warnings

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)

print('please select module: sm1, sm2, sm3, sm4')
dir_list = ('sm01', 'sm02', 'sm1', 'sm2', 'sm031', 'sm3', 'sm4')

chosen_dir = input()

while chosen_dir not in dir_list:
    print('Please enter valid module selection: sm1, sm2, sm3, sm4')
    chosen_dir = input()    




if chosen_dir == 'sm1':
    df1 = gl.data_loader(
        pth_1 = '/home/prethicktor/data/T_data/',
        pth_2 = '/home/prethicktor/data/RGI/rgi60-attribs/',
        pth_3 = '//home/prethicktor/data/matched_indexes/',
        pth_4 = '/home/prethicktor/data/regional_data/training_data/',
        RGI_input = 'n'
#                 scale = 'g',
#                 region_selection = 1,
#                 area_scrubber = 'off',
#                 anomaly_input = 5
    )
    dataset = df1
    dataset.name = 'df1'
    res = 'sr1'
#         print(module)
#         print(dataset)

if chosen_dir == 'sm2':
    df2 = gl.data_loader(
        pth_1 = '/home/prethicktor/data/T_data/',
        pth_2 = '/home/prethicktor/data/RGI/rgi60-attribs/',
        pth_3 = '//home/prethicktor/data/matched_indexes/',
        pth_4 = '/home/prethicktor/data/regional_data/training_data/',
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'off'
#                 anomaly_input = 5
    )
    dataset = df2
    dataset.name = 'df2'
    res = 'sr2'

if chosen_dir == 'sm3':
    df3 = gl.data_loader(
        pth_1 = '/home/prethicktor/data/T_data/',
        pth_2 = '/home/prethicktor/data/RGI/rgi60-attribs/',
        pth_3 = '//home/prethicktor/data/matched_indexes/',
        pth_4 = '/home/prethicktor/data/regional_data/training_data/',
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'on',
        anomaly_input = 1
    )
    dataset = df3
    dataset.name = 'df3'
    res = 'sr3'

if chosen_dir == 'sm4':
    df4 = gl.data_loader(
        pth_1 = '/home/prethicktor/data/T_data/',
        pth_2 = '/home/prethicktor/data/RGI/rgi60-attribs/',
        pth_3 = '//home/prethicktor/data/matched_indexes/',
        pth_4 = '/home/prethicktor/data/regional_data/training_data/',
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'on',
        anomaly_input = 5
    )
    dataset = df4
    dataset.name = 'df4'
    res = 'sr4'

deviations_1 = pd.read_csv('zults/deviations_' + dataset.name + '_1.csv')
deviations_2 = pd.read_csv('zults/deviations_' + dataset.name + '_0.csv')
deviations = pd.concat([deviations_1, deviations_2])
deviations = deviations.reset_index()
print('loading RGI...')
rootdir = '/data/fast0/datasets/rgi60-attribs/'
RGI_extra = pd.DataFrame()
for file in os.listdir(rootdir):
    file_reader = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
    RGI_extra = RGI_extra.append(file_reader, ignore_index = True)

    # select only RGI data that was used to train the model   
    RGI = RGI_extra[[
    'CenLat',
    'CenLon',
    'Slope',
    'Zmin',
    'Zmed',
    'Zmax',
    'Area',
    'Aspect',
    'Lmax'
    ]]

RGI = RGI.drop(RGI.loc[RGI['Zmed']<0].index)
RGI = RGI.drop(RGI.loc[RGI['Lmax']<0].index)
RGI = RGI.drop(RGI.loc[RGI['Slope']<0].index)
RGI = RGI.drop(RGI.loc[RGI['Aspect']<0].index)
RGI = RGI.reset_index()
RGI = RGI.drop('index', axis=1)
# RGI = RGI.rename(columns = {
# 'CenLon':'lon',
# 'CenLat':'lat',
# 'Area':'area',
# 'Slope':'mean_slope'
# })



if chosen_dir == 'sm1':
    RGI = RGI.rename(columns = {
        'CenLat':'Lat',
        'CenLon':'Lon',
        'Area':'Area',
        'Slope':'Mean Slope'
    })
    RGI = RGI[[
        'Lat',
        'Lon',
        'Area',
        'Mean Slope'
    ]]






if chosen_dir == 'sm5' or chosen_dir == 'sm6':
    print('loading RGI...')
    rootdir = '/data/fast1/glacierml/T_models/RGI/rgi60-attribs/'
    RGI_extra = pd.DataFrame()
    for file in tqdm(os.listdir(rootdir)):
        file_reader = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')

        # trim the RGIId entry to locate 2 digit region number.
        # Loop will only load desired RGI region based on these region tags
        region_1 = file_reader['RGIId'].iloc[-1][6:]
        region = region_1[:2]
        if str(region) == str(reg):
            RGI_extra = RGI_extra.append(file_reader, ignore_index = True)

    RGI = RGI_extra[[
        'CenLat',
        'CenLon',
        'Slope',
        'Zmin',
        'Zmed',
        'Zmax',
        'Area',
        'Aspect',
        'Lmax'
    ]]

    # here we want to drop any bad RGI data that can throw off predictions
    RGI = RGI.drop(RGI.loc[RGI['Zmed']<0].index)
    RGI = RGI.drop(RGI.loc[RGI['Lmax']<0].index)
    RGI = RGI.drop(RGI.loc[RGI['Slope']<0].index)
    RGI = RGI.drop(RGI.loc[RGI['Aspect']<0].index)
    RGI = RGI.reset_index()
    RGI = RGI.drop('index', axis=1)


deviations = deviations [[
'layer architecture',
'dropout',
# 'model parameters',
# 'total inputs',
'learning rate',
'epochs',
# 'test mae avg',
# 'train mae avg',
# 'test mae std dev',
# 'train mae std dev'
]]




print(deviations.to_string())
# here we can select an entry from the deviations table to make predictions. Default is top entry
print('Please select model index to predict thicknesses for RGI')
selected_model = int(input())
while type(selected_model) != int:
    print('Please select model index to predict thicknesses for RGI')
    selected_model = int(input()) 


arch = deviations['layer architecture'].loc[selected_model]
lr = deviations['learning rate'].loc[selected_model]
# vs = deviations['validation split'].iloc[selected_model]
ep = deviations['epochs'].loc[selected_model]
dropout = deviations['dropout'].loc[selected_model]
print('layer architecture: ' + arch + ' learning rate: ' + str(lr) + ' epochs: ' + str(ep))
print('predicting RGI thicknesses using model trained on RGI data matched with GlaThiDa thicknesses...')
dnn_model = {}
rootdir = 'saved_models/' + chosen_dir + '/'
RS = range(0,25,1)
dfs = pd.DataFrame()
for rs in tqdm(RS):
# each series is one random state of an ensemble of 25.
# predictions are made on each random state and appended to a df as a column
    model = (
        str(arch) +
        '_' +
        dataset.name +
        '_' +
        str(dropout) + 
        '_dnn_MULTI_' +
        str(lr) +
        '_' +
        str(0.2) +
        '_' +
        str(ep) + 
        '_' + 
        str(rs)
    )
    
    dnn_model[model] = tf.keras.models.load_model(
                    rootdir + 'sm_' + arch + '/' + 
                    dataset.name + 
                    '_' + 
                    str(dropout) + 
                    '_dnn_MULTI_' + 
                    str(lr) + 
                    '_' +
                    str(0.2) +
                    '_' +
                    str(ep) + 
                    '_' + 
                    str(rs)
    )
    
    s = pd.Series(
        dnn_model[model].predict(RGI, verbose=0).flatten(), 
        name = rs
    )

    dfs[rs] = s


# make a copy of RGI to add predicted thickness and their statistics
RGI_prethicked = RGI.copy() 
RGI_prethicked['avg predicted thickness'] = 'NaN'
RGI_prethicked['predicted thickness std dev'] = 'NaN'


print('calculating average thickness across random state ensemble...')
# loop through predictions df and find average across each ensemble of 25 random states
for i in tqdm(dfs.index):
    avg_predicted_thickness = np.mean(dfs.loc[i])
    RGI_prethicked['avg predicted thickness'].loc[i] = avg_predicted_thickness


print('computing standard deviations and variances for RGI predicted thicknesses')
# loop through predictions df and find std dev across each ensemble of 25 random states
for i in tqdm(dfs.index):


    predicted_thickness_std_dev = np.std(dfs.loc[i])
    RGI_prethicked['predicted thickness std dev'].loc[i] = predicted_thickness_std_dev

RGI_prethicked.to_csv(
    'zults/RGI_predicted_' +
    dataset.name + 
    '_' + 
    str(dropout) + 
    '_' + 
    arch + 
    '_' + 
    str(lr) + 
    '_' + 
    str(ep) + 
    '.csv'
)









