from transformers import AutoFeatureExtractor, ResNetForImageClassification, ResNetModel
from datasets import load_dataset
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import numpy as np
from copy import deepcopy
from torch.autograd import variable
from torch.utils.data import DataLoader
from data import load_wiki
from sls.adam_sls import AdamSLS
from sls.sgd_sls import SgdSLS
import wandb
from cosine_scheduler import CosineWarmupScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Image_classifier(nn.Module):

    def __init__(self,  num_classes, batch_size, args):
        super(Image_classifier, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.args = args
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")

        
        # for name,param in self.model.named_parameters():
        #     print(name)
        #out shape is (1,2048,7,7)
        self.fc1 = nn.Linear(2048,self.num_classes)
        self.pool = torch.nn.AvgPool2d(7)
        self.criterion = nn.CrossEntropyLoss()
        

        self.optimizer =[]
        if args.number_of_diff_lrs > 1:
            pparamalist = []
            for i in range(args.number_of_diff_lrs):
                paramlist = []
                optrangelower = math.ceil((4.0/(args.number_of_diff_lrs-2)) *(i-1))
                optrangeupper = math.ceil((4.0/(args.number_of_diff_lrs-2)) * (i))
                
                optrange = list(range(optrangelower,optrangeupper))
                if i == 0 or i == args.number_of_diff_lrs-1:
                    optrange =[]
                for name,param in self.named_parameters():
                    if "stages." in name:
                        included = False
                        for number in optrange:
                            if "stages." +str(number)+"." in name:
                                included = True
                        if included:
                            paramlist.append(param)
                        #      print("included", name , "in", i)
                    else:
                        if "embedder." in name:
                            if i == 0:
                                paramlist.append(param)
                            #      print("included", name , "in", i)
                        else:
                            if i == args.number_of_diff_lrs-1:
                                paramlist.append(param)
                                #   print("included", name , "in", i)
                if args.opts["opt"] == "adam":    
                    self.optimizer.append(optim.Adam(paramlist, lr=args.opts["lr"] ))
                if args.opts["opt"] == "sgd":    
                    self.optimizer.append(optim.SGD(paramlist, lr=args.opts["lr"] ))
                if args.opts["opt"] == "adamsls"  or args.opts["opt"] == "sgdsls":  
                    pparamalist.append(paramlist)
            if args.opts["opt"] == "adamsls":  
                self.optimizer.append(AdamSLS(pparamalist,strategy = args.update_rule ))
            if args.opts["opt"] == "sgdsls":  
                self.optimizer.append(SgdSLS(pparamalist ))
        else:
            if args.opts["opt"] == "adam":    
                self.optimizer.append(optim.Adam(self.parameters(), lr=args.opts["lr"] ))
            if args.opts["opt"] == "sgd":    
                self.optimizer.append(optim.SGD(self.parameters(), lr=args.opts["lr"] ))
            if args.opts["opt"] == "adamsls":    
                self.optimizer.append(AdamSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]] ,strategy = args.update_rule))
            if args.opts["opt"] == "sgdsls":    
                self.optimizer.append(SgdSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]]))

    def forward(self,x):
        x = self.model(**x).last_hidden_state
        x = self.pool(x).squeeze()
        return self.fc1(x)

    def fit(self,data, epochs, eval_ds = None):
        wandb.init(project="SLSforDifferentLayersImage"+self.args.ds, name = self.args.split_by + "_" + self.args.opts["opt"] + "_" + self.args.model +
        "_" + str(self.args.number_of_diff_lrs) +"_"+ self.args.savepth, entity="pkenneweg", 
        group = self.args.split_by + "_" + self.args.opts["opt"] + "_" + self.args.model +"_" + str(self.args.number_of_diff_lrs) )
        wandb.watch(self)
        
        self.mode = "cls"
        if (not self.args.opts["opt"] == "adamsls") and (not self.args.opts["opt"] == "sgdsls"):
            self.scheduler =[]
            for i in range(len(self.optimizer)): 
                self.scheduler.append(CosineWarmupScheduler(optimizer= self.optimizer[i], 
                                                warmup = math.ceil(len(data)*epochs *0.1 / self.batch_size) ,
                                                    max_iters = math.ceil(len(data)*epochs  / self.batch_size)))
        


        accuracy = None
        accsteps = 0
        accloss = 0
        for e in range(epochs):
            start = time.time()
            for index in range(0,len(data), self.batch_size):
                startsteptime = time.time()
                batch_x = data[index:index+self.batch_size]["img"]
                batch_y = torch.LongTensor(data[index:index+self.batch_size]["fine_label"]).to(device)
              #  print(batch_y.shape)
                batch_x = self.feature_extractor(batch_x, return_tensors="pt").to(device)

                if self.args.opts["opt"] == "adamsls" or self.args.opts["opt"] == "sgdsls":
                    closure = lambda : self.criterion(self(batch_x), batch_y)
               #     print(self(batch_x).shape)
                    for a in range(len(self.optimizer)):
                        self.optimizer[a].zero_grad()

                    for a in range(len(self.optimizer)):
                        loss = self.optimizer[a].step(closure = closure)
                else:
                    for a in range(len(self.optimizer)):
                        self.optimizer[a].zero_grad()
                    y_pred = self(batch_x)

                    loss = self.criterion(y_pred, batch_y)    
                    loss.backward()
                    for a in range(len(self.optimizer)):
                        self.optimizer[a].step()
                    for a in range(len(self.scheduler)):
                        self.scheduler[a].step()      

                dict = {"loss": loss.item() , "time_per_step":time.time()-startsteptime}    
                if self.args.opts["opt"] == "adamsls" or self.args.opts["opt"] == "sgdsls":
                   for a,step_size in enumerate( self.optimizer[0].state['step_sizes']):
                        dict["step_size"+str(a)] = step_size
                else:
                    for a,scheduler in enumerate( self.scheduler):
                        dict["step_size"+str(a)] = scheduler.get_last_lr()[0]
                    #      print(dict["step_size"+str(a)])
                wandb.log(dict)
                accloss = accloss + loss.item()
                accsteps += 1
               #if index % np.max((1,int((len(data)/self.batch_size)*0.1))) == 0:
                print(index, accloss/ accsteps)
                accsteps = 0
                accloss = 0
                if not eval_ds == None:
                    accuracy = self.evaluate(eval_ds)
                    wandb.log({"accuracy": accuracy})


    @torch.no_grad()
    def evaluate(self, data):
        resultx = None
        acc = 0
        for i in range(0,len(data), self.batch_size):
            batch_x = data[i:i+self.batch_size]["img"]
            batch_y = torch.LongTensor(data[i:i+self.batch_size]["fine_label"]).to(device)
            batch_x = self.feature_extractor(batch_x, return_tensors="pt").to(device)
            batch_x = self(batch_x)
            y_pred = torch.argmax(batch_x, dim = 1),
            accuracy = torch.sum(batch_y == y_pred)
            acc += accuracy
        
        acc = acc.item()/len(data)
        return acc


