import pandas as pd
import math


def init_data(df):

    df_train_sub = pd.read_csv('train_sub.csv', sep=',')    
    
    # combine all competitors data 
    comp_rate_list = ["comp"+str(i)+"_rate" for i in range(1,9)]
    comp_inv_list = ["comp"+str(i)+"_inv" for i in range(1,9)]
    comp_rate_percent_diff_list = ["comp"+str(i)+"_rate_percent_diff" for i in range(1,9)]
    for i  in range(1,9):
        df_train_sub["comp"+str(i)+"_rate_percent_diff_1"]= df_train_sub["comp"+str(i)+"_rate_percent_diff"]*df_train_sub["comp"+str(i)+"_rate"]
    comp_rate_percent_diff_list = ["comp"+str(i)+"_rate_percent_diff_1" for i in range(1,9)]
    df_train_sub['com_rate_mean'] = df_train_sub[comp_rate_list].mean(axis=1)
    df_train_sub['com_inv_mean'] = df_train_sub[comp_inv_list].mean(axis=1)
    df_train_sub['com_rate_percent_diff_min'] = df_train_sub[comp_rate_percent_diff_list].min(axis=1)
    df_train_sub['com_rate_percent_diff_max'] = df_train_sub[comp_rate_percent_diff_list].max(axis=1)
    df_train_sub = df_train_sub.loc[:, ~df_train_sub.columns.str.startswith('comp')]
    df_train_sub.drop('Unnamed: 0', axis=1, inplace = True)
    # price_usd normalized w.r.t srch_id
    df_train_sub['price_norm'] =df_train_sub[['price_usd', 'srch_id']].groupby('srch_id').transform(lambda x: (x-x.mean())/x.std())
    # Missing Values
    com_value = {'com_rate_mean':0 , 'com_inv_mean':0 ,
       'com_rate_percent_diff_min':0 , 'com_rate_percent_diff_max':0 }
    df_train_sub.fillna(value=com_value, inplace=True)

    df_train_sub['starrating_dif'] = abs(df_train_sub['visitor_hist_starrating'] - df_train_sub['prop_starrating'] )
    df_train_sub.fillna(value={'starrating_dif': 5}, inplace=True)
    df_train_sub['starrating_cat'] = pd.cut(df_train_sub['starrating_dif'], bins=[-1,1,4.99,df_train_sub['starrating_dif'].max()+1], labels=['match','not_match','no_info'])
    
    df_train_sub['hist_adr_usd'] = abs(df_train_sub['visitor_hist_adr_usd'] - df_train_sub['price_usd'])
    df_train_sub.fillna(value={'hist_adr_usd': -10}, inplace=True)
    df_train_sub['hist_adr_usd_cat'] = pd.cut(df_train_sub['hist_adr_usd'], bins=[-11,-1,50,df_train_sub['hist_adr_usd'].max()+1], labels=['no_info', 'match', 'not_match'])

    values = {'prop_review_score': float(-1), 'prop_location_score2': float(-1), 'srch_query_affinity_score': float(-90)}
    df_train_sub.fillna(value = values, inplace = True)
    
    df_final = df_train_sub.drop(['visitor_hist_starrating', 'visitor_hist_adr_usd', 'orig_destination_distance', 'gross_bookings_usd'], axis=1)
    
    return df_final

def evaluate (df):
    
    def get_dcg(df):
        dcg = sum([(2**rel-1)/math.log(i+2,2) for i, rel in enumerate(df["true"])])
        return dcg
    

    def nDCG (df):
        dcg = 0
        idcg = 0
        ndcg = 0
        df['true'] = 4*df["booking_bool"] + df["click_bool"]
        df1 = df.sort_values("pred",ascending=False)
        dcg = get_dcg(df1)
        df2 = df.sort_values("true",ascending=False)
        idcg = get_dcg(df2)
        ndcg = dcg/idcg
        return ndcg   

    return df.groupby('srch_id').apply(nDCG).mean()

def output_to_file(df, file):
    result = df.loc[:,["srch_id", "prop_id"]]
    result.to_csv(file,sep='\t', index=False)
    pass

def output_to_screen(matrix):
    print (" nDCG: ", matrix)
    pass