import pandas as pd
import util


class ML:
    def __init__(self):
        pass

    def preprocess(self, df):
        pass

    def prediction(self, df):
        pass

    def pipeline(self, df):
        df_preprocess = preprocess(df)
        df_pred= prediction(df_preprocess)
        eval_scores = util.evaluate(df_pred)
        return df_pred, eval_scores



