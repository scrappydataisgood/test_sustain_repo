import pandas as pd
import numpy as np
from functools import partial
from geneticalgorithm import geneticalgorithm as ga
from multiprocessing import Pool
import multiprocessing
import sklearn as skt
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import sklearn
import seaborn as sns
import pickle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
sns.set()
import matplotlib as mpl
mpl.rcParams.update({'figure.max_open_warning': 0})
import warnings
warnings.filterwarnings("ignore")


def run_ga_parallel():
    df = pd.read_csv("/mnt/CO2Pred/data/winter_dataset_filtered_by_outliers.csv").drop(columns = ['Unnamed: 0', 'Field_Name'])
    dfX = df.drop(columns = ["Yield_kg_per_Ha", "CO2_kg_per_Ha"])
    dfY = df[["Yield_kg_per_Ha", "CO2_kg_per_Ha"]]
    co2_max = df.CO2_kg_per_Ha.max()
    yield_max = df.Yield_kg_per_Ha.max()
    
    loaded_model_co2 = pickle.load(open('/mnt/CO2Pred/Pickle/predictor.pkl', 'rb'))
    loaded_standard_scaler_x_co2 = pickle.load(open('/mnt/CO2Pred/Pickle/X_Scalar.pkl', 'rb'))
    loaded_power_scaler_x_co2 = pickle.load(open('/mnt/CO2Pred/Pickle/X_Transformer.pkl', 'rb'))
    loaded_standard_scaler_y_co2 = pickle.load(open('/mnt/CO2Pred/Pickle/y_Scalar.pkl', 'rb'))
    loaded_power_scaler_y_co2 = pickle.load(open('/mnt/CO2Pred/Pickle/y_transformer.pkl', 'rb'))

    yield_models = []
    yield_x_scalar = []
    yield_x_trans = []
    yield_y_scalar = []
    yield_y_trans = []

    for i in range(1, 6):
        model_name= f'predictor_{i}{i}{i}.pkl'
        y_scalar = f'y_Scalar_{i}.pkl'
        x_scalar = f'X_Scalar_{i}.pkl'
        y_trans = f'y_transformer_{i}.pkl'
        x_trans = f'X_Transformer_{i}.pkl'

        loaded_model_yield = pickle.load(open(f'/mnt/CO2Pred/PickleEnsemble/{model_name}', 'rb'))
        loaded_standard_scaler_x_yield = pickle.load(open(f'/mnt/CO2Pred/PickleEnsemble/{x_scalar}', 'rb'))
        loaded_power_transformer_x_yield = pickle.load(open(f'/mnt/CO2Pred/PickleEnsemble/{x_trans}', 'rb'))
        loaded_standard_scaler_y_yield = pickle.load(open(f'/mnt/CO2Pred/PickleEnsemble/{y_scalar}', 'rb'))
        loaded_power_transformer_y_yield = pickle.load(open(f'/mnt/CO2Pred/PickleEnsemble/{y_trans}', 'rb'))

        yield_models.append(loaded_model_yield)
        yield_x_scalar.append(loaded_standard_scaler_x_yield)
        yield_x_trans.append(loaded_power_transformer_x_yield)
        yield_y_scalar.append(loaded_standard_scaler_y_yield)
        yield_y_trans.append(loaded_power_transformer_y_yield)
        
        algorithm_param = {'max_num_iteration': 40,\
                   'population_size':80,\
                   'mutation_probability':0.2,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.7,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':10}
        
    varbound=np.array([[dfX.Crop_Protection_Application_Doses.min(), dfX.Crop_Protection_Application_Doses.max()], 
    [dfX.Total_N.min(), dfX.Total_N.max()], 
    [dfX.Total_P.min(), dfX.Total_P.max()],  
    [dfX.Total_K.min(), dfX.Total_K.max()]])  
        
    num_processes = 7
    indices = np.linspace(0, len(dfX), num_processes + 1, dtype = int)
    starts = [[]]*num_processes
    for i in range(num_processes):
        starts[i] = [indices[i], indices[i+1]
    result = Pool(num_processes).map(run_partial, starts)
    return result

def run_partial(start_stop, varbound):    
     
                     
    start = start_stop[0]
    stop = start_stop[-1]
    
    prescription = {i:{"variable":None, "yield_co2_ratio": None, "yield_pred": None, "co2_pred":None} for i in range(start, stop)}
    opt_range = range(start, stop)
    alph = 0.1
    for index, row in dfX.iterrows():
        if index not in opt_range:
            continue
        context = row.to_numpy()[0]
        fun_to_optimize = partial(yield_co2_ratio, alph, context)

        model=ga(function=fun_to_optimize,dimension=4,\
             variable_type='real',\
             variable_boundaries=varbound,\
             algorithm_parameters=algorithm_param)
        model.run()

        prescription[index]['variable'] = model.output_dict['variable']
        prescription[index]['yield_co2_ratio'] = model.output_dict['function']

        full_features = np.hstack([context,model.output_dict['variable']]).reshape([1,-1])

        prescription[index]['yield_pred'] = predict_yield(full_features)
        prescription[index]['co2_pred'] = predict_co2(full_features)
    return prescription

def predict_yield(features):
    # TODO this will change when we get the hybrids
    total_sum_preds = 0
    for i in range(5):
        pred = yield_models[i].predict(yield_x_trans[i].transform(yield_x_scalar[i].transform(features))).reshape(-1,1)
        unscaled_pred = yield_y_scalar[i].inverse_transform(yield_y_trans[i].inverse_transform(pred))
        total_sum_preds += unscaled_pred
    return total_sum_preds / 5

def predict_co2(features):
    pred=loaded_model_co2.predict(loaded_power_scaler_x_co2.transform(loaded_standard_scaler_x_co2.transform(features))).reshape(-1,1)
    unscaled_pred = loaded_standard_scaler_y_co2.inverse_transform(loaded_power_scaler_y_co2.inverse_transform(pred))
    return unscaled_pred
                     
def yield_co2_ratio(alph, context, action):
    if alph > 1 or alph < 0:
        raise Exception("Illegal argument")
    """F are uncontrolled features, X are controlled features"""
    act = np.concatenate(([action[0]], [context], action[1:])).reshape([1,-1])
#     full_feat = np.hstack(action[0], context, action[1:]).reshape([1,-1])
    full_features = np.hstack([context,action]).reshape([1,-1])
    co2_pred = predict_co2(full_features)
    yield_pred = predict_yield(full_features)
    return alph*co2_pred/co2_max - (1-alph)*yield_pred/yield_max
#     return co2_pred / yield_pred                  

