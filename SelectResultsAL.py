import pandas as pd



file_name='AnomalyDetectionResults/IdealizedClasifitaction/results_10000sij_RUNS_801_945.txt'


frame=pd.read_csv(file_name)

frame=frame.sort_values(by=['auc_absbest'],ascending=False)

best_ones=frame.head(10)
print(best_ones)