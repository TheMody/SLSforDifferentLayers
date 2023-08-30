from transformers import ResNetModel, ResNetConfig, EfficientNetConfig, EfficientNetModel
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
from data import load_wiki
from sls.adam_sls import AdamSLS
from sls.Ken_sls import KenSLS
import wandb
from cosine_scheduler import CosineWarmupScheduler
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import ResNetModel

class Image_model(nn.Module):
    def __init__(self, model, out_shape, num_classes) -> None:
        super(Image_model, self).__init__()
        self.model = model
        self.fc1 = nn.Linear(out_shape,num_classes)

    def forward(self,x):
        # out = F.avg_pool2d(x, 4)
        # out = out.view(out.size(0), -1)
        x = self.model(x).last_hidden_state
   #     print(x.shape)
        mean = x.shape[2]*x.shape[3]
        x = torch.sum(x, dim = [2,3])/mean
        #x = self.pool(x).squeeze()
        return self.fc1(x)



class Image_trainer():

    def __init__(self,  num_classes, batch_size, args):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.args = args
       # self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        #self.model = ResNetModel.from_pretrained("microsoft/resnet-34")
        if args.model == "preeffNet":
        #    configuration = EfficientNetConfig()
            self.model = EfficientNetModel.from_pretrained("google/efficientnet-b7")
            out_shape = 768
        if args.model == "effNet":
            configuration = EfficientNetConfig()
            self.model = EfficientNetModel(configuration)
            out_shape = 2560
        
        if args.model == "resNet50":
            configuration = ResNetConfig() #this is resnet 50
            self.model = ResNetModel(configuration)
            out_shape = 2048
        if args.model == "preresNet34":
            self.model =  ResNetModel.from_pretrained("microsoft/resnet-34")
            out_shape = 512
        if args.model == "preresNet50":
            self.model =  ResNetModel.from_pretrained("microsoft/resnet-50")
            out_shape = 2048

        if args.model == "resNet34":
            # configuration = ResNetConfig(layer_type = "basic", hidden_sizes=[64, 128, 256, 512], depths = [3, 4, 6, 3]) #this is resnet 34
            # self.model = ResNetModel(configuration)
            self.model = ResNet34(num_classes).to(device)
            #out_shape = 512
        # elif args.model == "resNet50":
        #     self.model = ResNet50(num_classes).to(device)
        #     #out_shape = 512
        else:
            self.model = Image_model(self.model, out_shape, num_classes).to(device)
        self.model = torch.compile(self.model)

      #  self.preprocess_func = self.feature_extractor.preprocess
        # for name,param in self.model.named_parameters():
        #     print(name)
        #out shape is (1,2048,7,7)
        
        self.criterion = nn.CrossEntropyLoss()
        

        if args.number_of_diff_lrs > 1:
            pparamalist = []
            for i in range(args.number_of_diff_lrs):
                paramlist = []
                optrangelower = math.ceil((4.0/(args.number_of_diff_lrs-2)) *(i-1))
                optrangeupper = math.ceil((4.0/(args.number_of_diff_lrs-2)) * (i))
                
                optrange = list(range(optrangelower,optrangeupper))
                if i == 0 or i == args.number_of_diff_lrs-1:
                    optrange =[]
                for name,param in self.model.named_parameters():
                    if "stages." in name:
                        included = False
                        for number in optrange:
                            if "stages." +str(number)+"." in name:
                                included = True
                        if included:
                            paramlist.append(param)
                         #   print("included", name , "in", i)
                    else:
                        if "embedder." in name:
                            if i == 0:
                                paramlist.append(param)
                             #   print("included", name , "in", i)
                        else:
                            if i == args.number_of_diff_lrs-1:
                                paramlist.append(param)
                              #  print("included", name , "in", i)
                pparamalist.append(paramlist)
            if args.opts["opt"] == "adamsls":  
                    self.optimizer = AdamSLS(pparamalist,strategy = args.update_rule , combine_threshold = args.combine, c = self.args.c )
            if args.opts["opt"] == "sgdsls":  
                    self.optimizer = AdamSLS( pparamalist,strategy = args.update_rule, combine_threshold = args.combine, base_opt = "scalar",gv_option = "scalar", c = self.args.c  )
                 
        else:
           # print(args.opts["opt"])
            if args.opts["opt"] == "adam":    
                self.optimizer = optim.Adam(self.model.parameters(), lr=args.opts["lr"] )
            if args.opts["opt"] == "sgd":    
                self.optimizer = optim.SGD(self.model.parameters(), lr=args.opts["lr"] )
            if args.opts["opt"] == "kensls":    
                self.optimizer = KenSLS( [param for name,param in self.model.named_parameters() if not "pooler" in name] ,beta_s = self.args.beta, c = self.args.c)
            if args.opts["opt"] == "oladamsls":    
                self.optimizer = AdamSLS( [[param for name,param in self.model.named_parameters() if not "pooler" in name]] , c = 0.1, smooth = False )
            if args.opts["opt"] == "olsgdsls":    
                self.optimizer = AdamSLS( [[param for name,param in self.model.named_parameters() if not "pooler" in name]] , c = 0.1, base_opt = "scalar",gv_option = "scalar", smooth = False)
            if args.opts["opt"] == "adamsls":    
                self.optimizer = AdamSLS( [[param for name,param in self.model.named_parameters() if not "pooler" in name]] ,strategy = args.update_rule, combine_threshold = args.combine, c = self.args.c , beta_s = args.beta)
            if args.opts["opt"] == "sgdsls":    
                self.optimizer = AdamSLS( [[param for name,param in self.model.named_parameters() if not "pooler" in name]],strategy = args.update_rule, combine_threshold = args.combine, base_opt = "scalar",gv_option = "scalar", c = self.args.c , beta_s = args.beta )


    def fit(self,data, epochs, eval_ds = None, log_step =10):
        
        wandb.init(project="SLSforDifferentLayersImage_longer_new"+self.args.ds, name = self.args.split_by + "_" + self.args.opts["opt"] + "_" + self.args.model +
        "_" + str(self.args.number_of_diff_lrs) +"_"+ self.args.savepth, entity="pkenneweg", 
        group = "testresnet"+self.args.split_by + "_" + self.args.opts["opt"] + "_" + self.args.model +"_" + str(self.args.number_of_diff_lrs) + self.args.update_rule + str(self.args.combine)+"_c"+ str(self.args.c)+"_beta"+ str(self.args.beta) + "bs" + str(self.batch_size) )
        
        self.mode = "cls"
        if not "sls" in self.args.opts["opt"]:
            self.scheduler = CosineWarmupScheduler(optimizer= self.optimizer, 
                                                warmup = math.ceil(len(data)*epochs *0.1 ) ,
                                                    max_iters = math.ceil(len(data)*epochs  ))



        accuracy = None
        accsteps = 0
        accloss = 0
        for e in range(epochs):
           # if False:
          #  print(len(data))
            for index in range(len(data)):
                startsteptime = time.time()
                batch_x, batch_y = next(iter(data))
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
              #  print(batch_x.shape)
                if "sls" in self.args.opts["opt"]:
                    closure = lambda : self.criterion(self.model(batch_x), batch_y)
                    self.optimizer.zero_grad()
                    loss = self.optimizer.step(closure = closure)
                else:
                    self.optimizer.zero_grad()
                    y_pred = self.model(batch_x)

                    loss = self.criterion(y_pred, batch_y)    
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()      
               # print(self.args.opts["opt"])
                if index % log_step == 0:
                    dict = {"loss": loss.item() , "time_per_step":time.time()-startsteptime}    
                    if "sls" in  self.args.opts["opt"]:
                        if "kensls" in self.args.opts["opt"]:
                            dict["step_size0"] = self.optimizer.state["step_size"]
                            dict["loss_decrease"] = self.optimizer.state["loss_decrease"]
                            dict["gradient_norm"] = self.optimizer.state["gradient_norm"]
                            dict["c"] = self.optimizer.state["c"]
                            dict["average c"] = self.optimizer.state["average c"]
                        else:
                            for a,step_size in enumerate( self.optimizer.state['step_sizes']):
                                dict["step_size"+str(a)] = step_size
                                dict["avg_grad_norm"+str(a)] = self.optimizer.state["grad_norm_avg"][a]
                                dict["loss_decrease"+str(a)] = self.optimizer.state["loss_dec_avg"][a]
                    else:
                        dict["step_size"+str(0)] = self.scheduler.get_last_lr()[0]
                    wandb.log(dict)
                accloss = accloss + loss.item()
                accsteps += 1
                if index % np.max((1,int((len(data))*0.1))) == 0:
                    print(index*self.batch_size, accloss/ accsteps)
                    accsteps = 0
                    accloss = 0
            if not eval_ds == None:
                accuracy = self.evaluate(eval_ds)
                print("accuracy at epoch", e, accuracy)
                wandb.log({"accuracy": accuracy})
        wandb.finish()
        return accuracy

    @torch.no_grad()
    def evaluate(self, data):
        acc = 0
        for _ in range(len(data)):
            batch_x, batch_y = next(iter(data))
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x = self.model(batch_x)
            y_pred = torch.argmax(batch_x, dim = 1)
            accuracy = torch.sum(batch_y == y_pred)
            acc += accuracy
        
        acc = acc.item()/(len(data)*self.batch_size)
        return acc


