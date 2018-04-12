import pandas as pd
import util


class ML:
    def __init__(self):
        pass

    def preprocess(self, df):
		df['time'] = df['time'].astype(str).str[:-4]
		df['time'] = pd.to_datetime(df['time'], errors='coerce')
		df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
		
		#mood needs to be averaged
		df_mood = df.loc[df['variable'] == 'mood']
		df_mood = df_mood.groupby(['id', 'time'])['value'].mean().to_frame(name = 'mood_mean').reset_index()

		#others need to be summed
		grouped = df.groupby(['id','variable', 'time'])['value'].sum()
		grouped = grouped.unstack(level=[1]).reset_index()
		grouped.dropna(how='all', inplace=True)
		grouped = pd.concat([grouped, df_mood])
		grouped.drop('mood', axis=1, inplace=True)

		#NA to be filled with mean of sum
		'''still subject to change'''
		grouped.fillna(grouped.mean(), inplace=True)

    def prediction(self, df):
        pass

    def pipeline(self, df):
        df_preprocess = preprocess(df)
        df_pred= prediction(df_preprocess)
        eval_scores = util.evaluate(df_pred)
        return df_pred, eval_scores



