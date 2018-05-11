import pandas as pd
import util
import numpy as np

class Benchmark:
    def __init__(self):
        pass

    def prediction(self, df):
        ordinals = np.arange(len(df))
        df['pred'] = np.random.random(size=len(df))
        recommendations = df.loc[:,["srch_id", "prop_id","booking_bool","click_bool", 'pred' ]]
        return recommendations

    def pipeline(self, df):
        df_pred= self.prediction(df)
        eval_scores = util.evaluate(df_pred)
        return df_pred, eval_scores

