    import matplotlib.pyplot as plt
import pandas as pd

def PlotAUCwN(frame_all,model_dir):

    print(frame_all['num_events'])
    print(frame_all['auc_score'])
    
    plt.plot(frame_all['num_events'],frame_all['auc_score'],color='blue')
    plt.plot(frame_all['num_events'],frame_all['auc_score'],'.',color='blue')
    plt.xlabel('$N_{train}$')
    plt.ylabel('auc score')
    plt.xscale('log')
    plt.savefig(mother_dir+'/auc_events.png')
    

    return

mother_dir='/Users/humbertosmac/Documents/work/Transformers/Transformers_finke/test_results/classifier/Classification/'

frame_all=pd.read_csv(mother_dir+'/frame_auc.txt')
frame_all=frame_all.sort_values(by='num_events')
PlotAUCwN(frame_all,mother_dir)
