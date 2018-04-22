import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as select
import statsmodels.tsa.arima_model as stat
import statsmodels.graphics.tsaplots as tools
from pandas.tools.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
import pickle

warnings.filterwarnings('ignore')

def preprocess_mean(df,var):
    df.loc[df['variable'] == var]
    df['time'] = df['time'].astype(str).str[:-4]
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['just_date'] = df['time'].dt.date
    df_mood = df.loc[df['variable'] == var]
    df_preprocess = df_mood.groupby(['id', 'just_date'])['value'].mean().to_frame(name = '{0}_mean'.format(var)).reset_index()
    df_preprocess.set_index(['just_date'], drop=True, inplace=True)
    df_preprocess = df_preprocess.groupby(by='just_date').mean()
    df_preprocess = df_preprocess[2:]#drop the first 2 useless rows
    return df_preprocess

def preprocess_sum(df,var):
    df.loc[df['variable'] == var]
    df['time'] = df['time'].astype(str).str[:-4]
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['just_date'] = df['time'].dt.date
    df_mood = df.loc[df['variable'] == var]
    df_preprocess = df_mood.groupby(['id', 'just_date'])['value'].sum().to_frame(name = '{0}_sum'.format(var)).reset_index()
    df_preprocess.set_index(['just_date'], drop=True, inplace=True)
    df_preprocess = df_preprocess.groupby(by='just_date').sum()
    df_preprocess = df_preprocess.groupby(by='just_date').mean()
    df_preprocess = df_preprocess[2:]#drop the first 2 useless rows
    return df_preprocess

def preprocess(df, var, operation):
    if operation is 'mean':
        preprocess_it = preprocess_mean(df, var)
    if operation is 'sum':
        preprocess_it = preprocess_sum(df, var)
    if operation is 'max':
        preprocess_it = preprocess_max
    if operation is 'std':
        preprocess_it = preprocess_std
    if operation is 'min':
        preprocess_it = preprocess_min
    preprocess_it = preprocess_it.loc[pd.to_datetime('2014-03-04').date():pd.to_datetime('2014-06-08').date()]
    return preprocess_it

output_data = pd.DataFrame()
##preprocess


df1 = pd.read_csv('data/dataset_mood_smartphone.csv')
df1.drop('Unnamed: 0', axis=1, inplace=True)

id_list = df1['id'].unique()
print (id_list)

best_cfg_dict = {}

for individual in id_list:
    #individual = 'AS14.03' #for debugging
    df = df1
    df = df.loc[df['id'] == individual]
    #print (df['variable'].unique())


    a = preprocess_mean(df, 'mood')#sorry, this is stupid. but mood has to be preprocessed first:(
    #a[['mood_mean']].plot() #check if need differencing. plot shows no need since no trend of going up over time
    #features to be used: circumplex.arousal_mean; circumplex.valence_mean; activity_mean; entertainment_mean
    #print (a)
    var_fixed = ['circumplex.arousal', 'circumplex.valence', 'activity', 'appCat.entertainment']

    variables = df['variable'].unique()

    var_raw = []
    for var in variables:
            var_df = var+'_mean'
            var_raw.append(var_df)
            a[var_df] = preprocess(df, var, 'mean')
            #print (var_df, var)

    a['pred'] = a['mood_mean']
    a.fillna(a.mean(), inplace=True)
    a.dropna(axis=1, inplace=True)

    scaler = MinMaxScaler()
    scaler.fit(a)
    cols = a.columns.unique()
    a[cols] = scaler.transform(a[cols]) #standardize data

    '''
    blocked temporaly for testing

            for each in var_raw:
                    try:
                            a[['mood_mean', each]].plot()
                            plt.title('{0} of individual {1}'.format(each, individual))
                    except:
                            print ('{0} is not valid'.format(each))
                            continue
            plt.show()
# Graph shows no trend thus differencing term(d) is 0, but maybe seasonality. but could be ignored
'''

    length = len(a.index)
    split_point = round(length*0.75)# a manual split for train and test

    a['mood_mean'] = a['mood_mean'].apply(lambda x: float(x)) #some unknow error suggests only float accepted

    ex_terms = ['circumplex.arousal_mean', 'circumplex.valence_mean', 'activity_mean', 'appCat.entertainment_mean']#some pre-selected features that is correlated to target
    for each in ex_terms:
            if each not in a.columns:
                    ex_terms.remove(each)

    exog = a[ex_terms]
    endo = a['mood_mean']
    P = [1,2,4, 6]
    D = range(0, 2)
    Q = range(0, 2)
    a1 = a
    best_score, best_cfg, best_predict, best_aic= float('inf'), None, pd.DataFrame(), float('inf')
    for p in P:
         for d in D:
             for q in Q:
                orders = (p,d,q)
                exog = a[ex_terms]
                endo = a['mood_mean']
                try:
                    model = stat.ARIMA(endog=endo, exog=exog, freq='d', order=orders)#order is the parameter (p,d,q)
                    model_ = model.fit(disp=0)
                    a1['pred'] = model_.predict(start=split_point, exog=exog, dynamic=True)
                    aic = model_.aic
                    a1.fillna(a1.mean(), inplace=True)#fill Nan with mean first to inver_transform to original value
                    truth = a['mood_mean'].iloc[split_point:length]
                    predict = a1['pred'].iloc[split_point:length]
                    MSE = mean_squared_error(truth, predict)
                    if MSE < best_score:
                        best_predict = model_.predict(start=split_point, exog=exog, dynamic=True)
                        best_score, best_cfg, best_aic= MSE, orders, aic
                    print ('{3} arima{0} mse{1} aic{2}'. format(orders, MSE, aic, individual))
                except:
                    print ('arima{0} failed'. format(orders))
                    continue

    a1['mood_mean'] = a['mood_mean']
    try:
        a1['pred'] = best_predict
        a1.fillna(a1.mean(), inplace=True)#fill Nan with mean first to inver_transform to original value1['pred'] = best_predict
        a1[cols] = scaler.inverse_transform(a1[cols])
        a1['pred'].iloc[:split_point] = np.NaN
    except:
        continue
    #revert the dummy values to Nan so that they'll not be shown in the plot
    #a1[['mood_mean', 'pred']].plot()
    #plt.show()
    try:
        output_data[individual] = a1['pred']
        print ('{0} Best ARIMA{1}s MSE={2} AIC={3}'.format(individual, best_cfg, best_score, best_aic))
    except:
        print ('{0} has no prediction'.format(individual))
    print (output_data.tail(1))

    best_cfg_dict[individual] = best_cfg

print('finish tuning, and save the best configure parameters')
print(best_cfg_dict)
pickle.dump(best_cfg_dict, open('output/arima_best_cfg.pkl', 'wb'))

# output_data.to_csv('prediction.csv')
# plt.show()
