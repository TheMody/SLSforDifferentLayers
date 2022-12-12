
from datasets import load_dataset
from image_classifier import Image_classifier
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_img(args,config):
    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]
    optimizer = config["DEFAULT"]["optim"]

    args.number_of_diff_lrs = int(config["DEFAULT"]["num_diff_opt"])
    args.opts = {"lr": 1e-4, "opt": config["DEFAULT"]["optim"]}
    args.ds = dataset
    args.split_by = config["DEFAULT"]["split_by"]
    args.update_rule = config["DEFAULT"]["update_rule"]
    args.model = config["DEFAULT"]["model"]
    args.savepth = config["DEFAULT"]["directory"]
    args.combine = float(config["DEFAULT"]["combine"])


    dataset = load_dataset("cifar100")
  #  image = dataset["test"]["image"][0]

    img_cls = Image_classifier(100,batch_size,args).to(device)
    
    img_cls.fit(dataset["train"],max_epochs, dataset["test"])
    # model predicts one of the 1000 ImageNet classes
    #predicted_label = logits.argmax(-1).item()
    #print(img_cls.model.config.id2label[predicted_label])