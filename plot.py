

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd 
import wandb




#print(input)

def log_average_weighted(input, ranking,  base = 1.8):
    weighted_average = np.zeros(input[0].shape)
    sum = 0
    for a,i in enumerate(ranking):
        element = input[i]
        element2 = np.asarray([math.log(a) for a in element])
        weighted_average += element2 * math.pow(base, -1*a) /2
        sum += math.pow(base, -1*a) /2
    weighted_average = weighted_average / sum
    weighted_average = np.asarray([math.exp(a) for a in weighted_average])
    
    return weighted_average

def log_average(input):
    weighted_average = np.zeros(input[0].shape)
    sum = 0
    for element in input:
        element2 = np.asarray([math.log(a) for a in element])
        weighted_average += element2 
        sum += 1
    weighted_average = weighted_average / sum
    weighted_average = np.asarray([math.exp(a) for a in weighted_average])
    
    return weighted_average

def process_log(input):
    accuracies1 = []
    accuracies2 = []
    accuracies3 = []
    lrs =[]
    lr = []
    max_lrs = 10
    for i,element in enumerate(input):
        #print(i, element)
        if (i +13) % 13 == 0 :
             accuracies1.append(element)
        elif (i +12) % 13 == 0 :
             accuracies2.append(element)
        elif (i +11) % 13 == 0 :
             accuracies3.append(element)
        else:
            lr.append(element)
            if len(lr) >= 10:
                lrs.append(lr)
                lr = []
                
    lrs = np.asarray(lrs)
    lrs = lrs[:,:max_lrs]
    
    rating = np.asarray(accuracies2) + np.asarray(accuracies3)
#     print(rating)
    ranking = np.argsort(rating)
    ranking = np.flip(ranking)
    print(ranking)
    
    
    return ranking, accuracies1,  accuracies2, accuracies3, lrs

def plot_lr_rate(input):
    base = 1.8
    ranking, accuracies1,  accuracies2, accuracies3, lrs = process_log(input)
    weighted_average = log_average_weighted(lrs, ranking, base = base)
    weighted_var = np.zeros(lrs[0].shape)
   # for i,element in enumerate(lrs):#
    
  #  for i,element in enumerate(lrs):
    sum = 0
    for a,i in enumerate(ranking):
        element = lrs[i]
      #  element = np.asarray([math.log(a) for a in element])
        weighted_var += np.asarray([math.log(a) for a in abs(weighted_average-element)]) * math.pow(base, -1* a) /2
      #  weighted_var += weighted_average-element * math.pow(2, -1* ranking[i]) /2
        sum += math.pow(base, -1*a) /2
    weighted_var = weighted_var / sum
    weighted_var = np.asarray([math.exp(a) for a in weighted_var])
    print(weighted_average)
    print(weighted_var)
    print(lrs[ranking[0]])
    
    x = [i for i in range(lrs[0].shape[0])]
    plt.plot(x,weighted_average, color = (0,0.1,1), label = "weighted average")
    plt.fill_between(x, [max(a,1e-7) for a in weighted_average-weighted_var], weighted_average+weighted_var, color = (0,0.1,1, 0.3), label = "deviation")
    plt.plot(x,lrs[ranking[0]], label = "best")
    plt.legend()
    plt.yscale('log')
    # for i,lr in enumerate(lrs):
    #     plt.plot(lr,color = (1,0,0, 1-ranking[i]/ranking.shape[0]))
    
    plt.show()

def combine_multiple(inputs):
    averages =[]
    for input in inputs:
        ranking, accuracies1,  accuracies2, accuracies3, lrs = process_log(input)
        weighted_average = log_average_weighted(lrs, ranking, base = 1.8)
        x = [i for i in range(weighted_average.shape[0])]
  #      plt.plot(x,weighted_average)
        averages.append(weighted_average)
    average = log_average(averages)
    
    print("average over all results",np.asarray(average))
    
    plt.plot(x,[2e-5 for a in x], label = "baseline")
    plt.plot(x,average, color = (0,0.1,1), label = "combined learning rate")
    plt.legend()
    plt.xlabel("choice")
    plt.ylabel("learning rate")
    plt.yscale('log')
    plt.show()

def plot_TSNE_clustering(X,y):
   # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    X_embedded = TSNE(n_components=2,init='random').fit_transform(X)
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c = y)
    plt.show()

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
                        if i >= len(colortable): 
                            i = 0
                        print(i)
                        c = list(colors.to_rgba(colortable[i]))
                        c[3] = c[3]*0.3
                        c = tuple(c)
                        plt.fill_between(np.arange(len(mean)), mean + error, mean - error, color = c)
                        plt.plot(mean, color = colortable[i])
                        i = i +1
                        
                        legendlist = prev_group
                    acc_loss = []
                    acc_accuracy = []
                    prev_group = run.group
                acc_loss.append(run.history()["loss"][run.history()["loss"].isnull() == False].to_numpy())
                acc_accuracy.append( run.history()["accuracy"][run.history()["accuracy"].isnull() == False].to_numpy()) 
    plt.legend(legendlist)
    plt.show()   
           # print(run.history()["loss"][run.history()["loss"].isnull() == False])
            # for i, row in run.history().iterrows():
            #     print(row["_timestamp"], row["loss"])
                #print(run)































