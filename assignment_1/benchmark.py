import pandas as pd
import util


class Benchmark:
    def __init__(self):
        pass

    def preprocess(self, df):
        df.loc[df['variable'] == "mood"]
        df['time'] = df['time'].astype(str).str[:-4]
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['just_date'] = df['time'].dt.date
        df_mood = df.loc[df['variable'] == "mood"]
        df_preprocess = df_mood.groupby(['id', 'just_date'])['value'].mean().to_frame(name = 'mood_mean').reset_index()
        df_preprocess['label'] = df_preprocess['mood_mean']
        return df_preprocess

    def prediction(self, df):
        df['pred'] = df['mood_mean'].shift(1)
        return df

    def pipeline(self, df):
        df_preprocess = self.preprocess(df)
        df_pred= self.prediction(df_preprocess)
        eval_scores = util.evaluate(df_pred)
        return df_pred, eval_scores



