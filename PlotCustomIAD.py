import matplotlib.pyplot as plt
import pandas as pd


def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier

def PlotTopAUC(fpr,tpr,run,auc,max_sic):
    plt.plot(tpr,1/fpr, label='RUN='+str(run)+' AUC='+str(truncate_float(auc,3))+' maxSIC='+str(truncate_float(max_sic,3)),linestyle='--') #color=color)
    plt.ylim(1, 1e5)
    plt.xlim(0, 1)
    plt.xlabel(r"$ tpr$")
    plt.ylabel(r"$1 / fpr$")
    plt.yscale('log')
    plt.legend()

    return

main_dir='./AnomalyDetectionResults/Supervised/'
file_name=main_dir+/''

print(top_runs)
plot_title='CLS head - supervised '
plt.legend()
plt.title(plot_title)
plt.savefig(main_dir+'/roc_curve_RUNS_'+str(first_run)+'_'+str(last_run)+'_custom.png')
plt.close()