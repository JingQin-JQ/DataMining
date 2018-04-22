import pandas as pd
import util


class Benchmark:
    def __init__(self):
        pass

    def preprocess(self, df):
        df.rename(columns={'mood':'label'}, inplace=True)
        df['pred'] = df['label'].shift(1)
        df = df[~pd.isnull(df['label'])]
        return df

    def prediction(self, df):
        return df

    def pipeline(self, df):
        df_preprocess = self.preprocess(df)
        df_train = df_preprocess.groupby('id').apply(util.get_trainset).reset_index().drop('level_1', axis=1)
        df_test= df_preprocess.groupby('id').apply(util.get_testset).reset_index().drop('level_1', axis=1)
        df_pred= self.prediction(df_test)
        eval_scores = util.evaluate(df_pred)
        return df_pred, eval_scores



