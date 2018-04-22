import pandas as pd
from sklearn import preprocessing
import statsmodels.tsa.stattools as select
import statsmodels.tsa.arima_model as stat
import statsmodels.graphics.tsaplots as tools
import pickle
import util


class Temporal:
    def __init__(self):
        pass

    def preprocess(self, df):
        def fill_circumplex_na(df, fill_val):
            df['circumplex.arousal'] = df['circumplex.arousal'].fillna(fill_val)
            df['circumplex.valence'] = df['circumplex.valence'].fillna(fill_val)
            return df

        def fillna_func(df, col_l_minmax):
            #df[col_l_minmax] = df[col_l_minmax].apply(lambda x: x.fillna(x.mean()),axis=0)
            df[col_l_minmax] = df[col_l_minmax].fillna(method='ffill')
            df[col_l_minmax] = df[col_l_minmax].fillna(0)

            df = fill_circumplex_na(df, 0)
            return df

        def generate_label(df):
            df['label'] = df['mood'].fillna(method='ffill')
            return df

        def remove_nan_label(df):
            return df[~pd.isnull(df['label'])]

        def min_max_norm_func(df, col_l_minmax):
            # min max scaler
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(df[col_l_minmax])
            df_norm = pd.DataFrame(x_scaled, columns=col_l_minmax, index=df.index)

            col_origin = ['id', 'date', 'mood', 'weekday', 'circumplex.arousal', 'circumplex.valence', 'label']
            df_norm = pd.concat((df_norm, df[col_origin]), axis=1)

            return df_norm

        def prepare_df(df, col_l_minmax):
            df = df.sort_values(by='date')
            df = fillna_func(df, col_l_minmax)
            df = generate_label(df)
            df = remove_nan_label(df)
            df = min_max_norm_func(df, col_l_minmax)
            return df

        col_l_minmax = ['appCat.builtin', 'appCat.communication', 'appCat.entertainment',
            'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other',
            'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities',
            'appCat.weather', 'call', 'screen', 'sms', 'activity',
                'appCat.fun_total', 'appCat.Serious_total']

        df_preprocess = df.groupby('id')\
                          .apply(lambda x: prepare_df(x, col_l_minmax))\
                          .drop('id', axis=1).reset_index().drop('level_1', axis=1)
        return df_preprocess

    def train_prediction(self, df):
        test_ratio = 0.25
        orders_best_cfg = pickle.load(open('output/arima_best_cfg.pkl', 'rb'))
        #some pre-selected features that is correlated to target
        ex_terms = ['circumplex.arousal', 'circumplex.valence', 'activity', 'appCat.entertainment']

        df_pred_l = []

        for people, df_sub in list(df.groupby('id')):
            df_pred = pd.DataFrame()

            split_point = round(len(df_sub)*(1-test_ratio))
            exog = df_sub[ex_terms].values
            endo = df_sub['label'].values
            orders = orders_best_cfg.get(people, (4, 0, 1))

            try:
                model = stat.ARIMA(endog=endo, exog=exog, freq='d', order=orders)
                model_ = model.fit(disp=0)
            except:
                model = stat.ARIMA(endog=endo, exog=exog, freq='d', order=(4, 0, 1))
                model_ = model.fit(disp=0)

            df_pred['pred'] = model_.predict(start=split_point, exog=exog, dynamic=True)
            df_pred['label'] = df_sub['label'].iloc[split_point:].values
            df_pred['date'] = df_sub['date'].iloc[split_point:].values
            df_pred['id'] = people

            df_pred_l.append(df_pred)

        df_pred = pd.concat(df_pred_l)
        return df_pred

    def pipeline(self, df):
        df_preprocess = self.preprocess(df)
        df_pred= self.train_prediction(df_preprocess)
        eval_scores = util.evaluate(df_pred)
        return df_pred, eval_scores



