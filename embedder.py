from transformers import BertTokenizer, BertModel, ElectraTokenizer, ElectraModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import numpy as np
from transformers.utils import logging
from transformers import glue_convert_examples_to_features, DataCollatorForLanguageModeling
from copy import deepcopy
from torch.autograd import variable
from torch.utils.data import DataLoader
from data import load_wiki
import os
from sls.adam_sls import AdamSLS
from sls.sgd_sls import SgdSLS
import wandb
logging.set_verbosity_error()


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    


    


class NLP_embedder(nn.Module):

    def __init__(self,  num_classes, batch_size, args):
        super(NLP_embedder, self).__init__()
        self.type = 'nn'
        self.batch_size = batch_size
        self.padding = True
        self.bag = False
        self.num_classes = num_classes
        self.lasthiddenstate = 0
        self.args = args

        if args.model == "bert":
            from transformers import BertTokenizer, BertForMaskedLM
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.output_length = 768
        if args.model == "roberta":
            from transformers import RobertaTokenizer, RobertaModel
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaModel.from_pretrained('roberta-base')
            self.output_length = 768

        
        self.fc1 = nn.Linear(self.output_length,self.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        

        self.optimizer =[]
        if args.split_by == "layer":
            if args.number_of_diff_lrs > 1:
                pparamalist = []
                for i in range(args.number_of_diff_lrs):
                    paramlist = []
                    optrangelower = math.ceil((12.0/(args.number_of_diff_lrs-2)) *(i-1))
                    optrangeupper = math.ceil((12.0/(args.number_of_diff_lrs-2)) * (i))
                    
                    optrange = list(range(optrangelower,optrangeupper))
                    if i == 0 or i == args.number_of_diff_lrs-1:
                        optrange =[]
                    for name,param in self.named_parameters():
                        if "encoder.layer." in name:
                            included = False
                            for number in optrange:
                                if "." +str(number)+"." in name:
                                    included = True
                            if included:
                                paramlist.append(param)
                              #  print("included", name , "in", i)
                        else:
                            if "embeddings." in name:
                                if i == 0:
                                    paramlist.append(param)
                                 #   print("included", name , "in", i)
                            else:
                                if i == args.number_of_diff_lrs-1 and not "pooler" in name:
                                    paramlist.append(param)
                                  #  print("included", name , "in", i)
                                  #  print(name, param.requires_grad, param.grad)
                    if args.opts["opt"] == "adam":    
                        self.optimizer.append(optim.Adam(paramlist, lr=args.opts["lr"] ))
                    if args.opts["opt"] == "sgd":    
                        self.optimizer.append(optim.SGD(paramlist, lr=args.opts["lr"] ))
                    if args.opts["opt"] == "adamsls"  or args.opts["opt"] == "sgdsls":  
                        pparamalist.append(paramlist)
                if args.opts["opt"] == "adamsls":  
                    self.optimizer.append(AdamSLS(pparamalist))
                if args.opts["opt"] == "sgdsls":  
                    self.optimizer.append(SgdSLS(pparamalist ))
            else:
                if args.opts["opt"] == "adam":    
                    self.optimizer.append(optim.Adam(self.parameters(), lr=args.opts["lr"] ))
                if args.opts["opt"] == "sgd":    
                    self.optimizer.append(optim.SGD(self.parameters(), lr=args.opts["lr"] ))
                if args.opts["opt"] == "adamsls":    
                    self.optimizer.append(AdamSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]]))
                if args.opts["opt"] == "sgdsls":    
                    self.optimizer.append(SgdSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]]))
        else:
            querylist = []
            keylist = []
            valuelist = []
            elselist = [] 
            for name,param in self.named_parameters():
                if not "pooler" in name:
                    if "query" in name:
                        querylist.append(param)
                    else:
                        if "key" in name:
                            keylist.append(param)
                        else:
                            if "value" in name:
                                valuelist.append(param)
                            else:
                                elselist.append(param)
            if args.opts["opt"] == "adam":    
                self.optimizer.append(optim.Adam(self.parameters(), lr=args.opts["lr"] ))
            if args.opts["opt"] == "sgd":    
                self.optimizer.append(optim.SGD(self.parameters(), lr=args.opts["lr"] ))
            if args.opts["opt"] == "adamsls":    
                self.optimizer.append(AdamSLS( [keylist,querylist,valuelist,elselist]))
            if args.opts["opt"] == "sgdsls":    
                self.optimizer.append(SgdSLS( [keylist,querylist,valuelist,elselist]))
            

            
        
    def forward(self, x_in):
        x = self.model(**x_in).last_hidden_state
        x = x[:, self.lasthiddenstate]
        x = self.fc1(x)
        return x
    
    
     
    def fit(self, x, y, epochs=1, X_val= None,Y_val= None):
        wandb.init(project="SLSforDifferentLayers"+self.args.ds, name = self.args.split_by + "_" + self.args.opts["opt"] + "_" + self.args.model +
        "_" + str(self.args.number_of_diff_lrs) +"_"+ self.args.savepth)
        wandb.watch(self)
        
        
        if (not self.args.opts["opt"] == "adamsls") and (not self.args.opts["opt"] == "sgdsls"):
            self.scheduler =[]
            for i in range(len(self.optimizer)): 
                self.scheduler.append(CosineWarmupScheduler(optimizer= self.optimizer[i], 
                                                warmup = math.ceil(len(x)*epochs *0.1 / self.batch_size) ,
                                                    max_iters = math.ceil(len(x)*epochs  / self.batch_size)))
        


        accuracy = None
        accsteps = 0
        accloss = 0
        for e in range(epochs):
            start = time.time()
            for i in range(math.ceil(len(x) / self.batch_size)):
                startsteptime = time.time()
                ul = min((i+1) * self.batch_size, len(x))
                batch_x = x[i*self.batch_size: ul]
                batch_y = y[i*self.batch_size: ul]
                batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                batch_y = batch_y.to(device)
                batch_x = batch_x.to(device)

                if self.args.opts["opt"] == "adamsls" or self.args.opts["opt"] == "sgdsls":
                    closure = lambda : self.criterion(self(batch_x), batch_y)

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
                if i % np.max((1,int((len(x)/self.batch_size)*0.1))) == 0:
                    print(i, accloss/ accsteps)
                    accsteps = 0
                    accloss = 0

            if X_val != None:
                with torch.no_grad():
                    accuracy = self.evaluate(X_val, Y_val).item()
                    print("accuracy after", e, "epochs:",accuracy, "time per epoch", time.time()-start)
                    wandb.log({"accuracy": accuracy})
            else:
                print("epoch",e,"time per epoch", time.time()-start)
                
                
        wandb.finish()
        return accuracy
        
    @torch.no_grad()
    def evaluate(self, X,Y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Y = Y.to(device)
        y_pred = self.predict(X)
        accuracy = torch.sum(Y == y_pred)
        accuracy = accuracy/Y.shape[0]
        return accuracy
    
    def predict(self, x):
        resultx = None

        for i in range(math.ceil(len(x) / self.batch_size)):
            ul = min((i+1) * self.batch_size, len(x))
            batch_x = x[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_x = batch_x.to(device)
            batch_x = self(batch_x)
            if resultx is None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx, batch_x.detach()))

     #   resultx = resultx.detach()
        return torch.argmax(resultx, dim = 1)
    
    def embed(self, x):
        resultx = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(math.ceil(len(x) / self.batch_size)):
            ul = min((i+1) * self.batch_size, len(x))
            batch_x = x[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
            batch_x = batch_x.to(device)
            batch_x = self.model(**batch_x,output_hidden_states = True)   
            batch_x = batch_x.hidden_states[-1]
            batch_x = batch_x[:, self.lasthiddenstate]
            if resultx is None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx, batch_x.detach()))

     #   resultx = resultx.detach()
        return resultx
    
    

        
        

