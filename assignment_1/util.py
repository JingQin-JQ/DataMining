def evaluate(df_pred):
    RMSE = ((df_pred.label - df_pred.pred) ** 2).mean() ** .5
    return RMSE

def output_to_file(predictions, evaluation_scores):
    pass

def output_to_screen(matrix):
    print (matrix)
    pass