

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd 
import wandb



def smoothing(list, length = 10):
    return [np.mean(list[max(i-length, 0 ): i]) for i in range(len(list))]
    
if __name__ == '__main__': 
    api = wandb.Api()
    entity, project = "pkenneweg", "SLSforDifferentLayersqnlismall"  # set to your entity and project 
    runs = api.runs(entity + "/" + project) 
    prev_group = ""
    acc_loss = []
    acc_accuracy = []
    i = 0
    colortable = list(colors.TABLEAU_COLORS.values())
   # print(colortable)
    legendlist = []
    for run in runs:
        if run.state == "finished":
            if "cycle" in run.name or "impact_mag" in run.name:
                print(run.name)
                if not run.group == prev_group:
                    if len(acc_loss) > 4:
                        acc_loss = [smoothing(a) for a in acc_loss]
                        mean = np.mean(acc_loss, axis = 0)
                        std = np.std(acc_loss, axis = 0)
                        error = std/np.sqrt(len(acc_loss))
                        c = list(colors.to_rgba(colortable[i]))
                        c[3] = c[3]*0.3
                        c = tuple(c)
                        plt.fill_between(np.arange(len(mean)), mean + error, mean - error, color = c)
                        plt.plot(mean, color = colortable[i])
                        i = i +1
                        
                        legendlist.append( prev_group)
                    acc_loss = []
                    acc_accuracy = []
                    prev_group = run.group
                acc_loss.append(run.history()["loss"][run.history()["loss"].isnull() == False].to_numpy())
                acc_accuracy.append( run.history()["accuracy"][run.history()["accuracy"].isnull() == False].to_numpy()) 
    plt.legend(legendlist)
    plt.show()   
































