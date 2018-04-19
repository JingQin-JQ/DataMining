import pandas as pd
from pandas.tseries.offsets import DateOffset

def get_cutoff_date(df, test_ratio=0.1):
    
    end_date = max(df.date)
    start_date = min(df.date)
    nr_days = (end_date - start_date).days
    
    test_data_days = int(nr_days * test_ratio)
    
    cutoff_date = end_date - DateOffset(days=test_data_days)
    
    return cutoff_date

def get_trainset (df, test_ratio=0.1):
    df = df.drop(["id"], axis=1)
    cutoff_date = get_cutoff_date(df)
    df_train = df[df['date'] < cutoff_date]
    return df_train


def get_testset (df, test_ratio=0.1):
    df = df.drop(["id"], axis=1)
    cutoff_date = get_cutoff_date(df)
    df_test = df[df['date'] >= cutoff_date]
    return df_test


def fill_gap(df):
    return df.reset_index()\
            .groupby('id')\
            .apply(lambda g: g.reset_index().drop('id', axis=1).set_index('date').asfreq('D'))\
            .reset_index()


def add_extra_features(df):
    df['weekday'] = df.index.get_level_values('date').dayofweek
    df['appCat.fun_total'] = df['appCat.communication'] + \
                             df['appCat.entertainment'] + \
                             df['appCat.game'] + \
                             df['appCat.social'] + \
                             df['appCat.travel']
    df['appCat.Serious_total'] = df['appCat.finance']+df['appCat.office']
    return df


def init_data(df):

    data = df.drop('Unnamed: 0', axis=1)
    # get date from time 
    data['time'] = pd.to_datetime(data['time'], errors='coerce')
    data['date'] = data['time'].dt.date
    # sum up all Duration and count type feature 
    list_sum = ['screen', 'call', 'sms', 'appCat.builtin', 'appCat.communication',
       'appCat.entertainment', 'appCat.finance', 'appCat.game',
       'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
       'appCat.unknown', 'appCat.utilities', 'appCat.weather']
    df_sum = data.loc[data['variable'].isin(list_sum)]\
             .pivot_table(index=['id','date'], columns='variable', values='value', aggfunc='sum')
    # get mean of all score type feature     
    list_mean = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    df_mean = data.loc[data['variable'].isin(list_mean)]\
              .pivot_table(index=['id','date'], columns='variable', values='value', aggfunc='mean')
    # join together all features
    df = df_sum.join(df_mean, how='outer')
    # add new features
    df = add_extra_features(df)
    # fill gaps with NaN
    df = fill_gap(df)
    
    return df

 

def evaluate(df_pred):
    RMSE = ((df_pred.label - df_pred.pred) ** 2).mean() ** .5
    return RMSE

def output_to_file(df, file):
    result = df.loc[:,['id','label','pred']]
    result.to_csv(file)
    pass

def output_to_screen(matrix):
    print (" RMSE: ", matrix)
    pass