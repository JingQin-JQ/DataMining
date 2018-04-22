import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
import lightgbm as lgb

import util

class ML:
    def __init__(self):
        pass



    def preprocess(self, df):
        def avg_features(df, nr_days=5):
            # set date as index
            df = df.drop('id', axis=1).set_index('date')

            # get average value of last 5 days
            df_mean = df.rolling(window=nr_days, min_periods=1).mean()

            # use 6th day as lable
            df_mean['label'] = df['mood'].shift(-1)
            df_mean = df_mean.drop('mood', axis=1)

            # remove samples without target (mood)
            df_mean = df_mean[~pd.isnull(df_mean['label'])]

            return df_mean

        df = df.reset_index()\
               .groupby('id')\
               .apply(avg_features)\
               .drop(['level_0', 'index'], axis=1)\
               .reset_index()
        return df

    def model(self, df):
        X_train = df.drop(['id', 'date','label'], axis=1)
        y_train = df['label']
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.05]}

        lgb_estimator = lgb.LGBMRegressor(boosting_type='gbdt',
                                  objective='regression',
                                  learning_rate=0.01)


        gsearch = GridSearchCV(estimator=lgb_estimator,
                       param_grid=param_grid,
                       scoring="mean_squared_error",
                       cv=10,
                       verbose=1)

        lgb_model = gsearch.fit(X_train.values, y_train.values)
        return lgb_model

    def prediction(self, model, df):
        X_test = df.drop(['id', 'date','label'], axis=1)
        df['pred'] = model.predict(X_test)
        return df


    def pipeline(self, df):
        df_preprocess = self.preprocess(df)
        df_train = df_preprocess.groupby('id').apply(util.get_trainset).reset_index().drop('level_1', axis=1)
        df_test= df_preprocess.groupby('id').apply(util.get_testset).reset_index().drop('level_1', axis=1)
        model = self.model(df_train)
        df_pred= self.prediction(model, df_test)
        eval_scores = util.evaluate(df_pred)
        return df_pred, eval_scores



