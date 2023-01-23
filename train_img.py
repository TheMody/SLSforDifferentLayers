
from datasets import load_dataset
from image_classifier import Image_classifier
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_img(args,config):
    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]

    args.number_of_diff_lrs = int(config["DEFAULT"]["num_diff_opt"])
    args.opts = {"lr": 1e-4, "opt": config["DEFAULT"]["optim"]}
    args.ds = dataset
    args.split_by = config["DEFAULT"]["split_by"]
    args.update_rule = config["DEFAULT"]["update_rule"]
    args.model = config["DEFAULT"]["model"]
    args.savepth = config["DEFAULT"]["directory"]
    args.combine = float(config["DEFAULT"]["combine"])

    if dataset == "cifar100":
      num_classes = 100
      labelname = "fine_label"
      dataname = "img"
    if dataset == "cifar10":
      num_classes = 10
      labelname = "label"
      dataname = "img"
    if dataset == "mnist":
      num_classes = 10
      labelname = "label"
      dataname = "image"
    img_cls = Image_classifier(num_classes,batch_size,args).to(device)
    dataset = load_dataset(dataset)
    img_cls.fit(dataset["train"],max_epochs, dataset["test"], labelname = labelname, dataname = dataname)
    # model predicts one of the 1000 ImageNet classes
    #predicted_label = logits.argmax(-1).item()
    #print(img_cls.model.config.id2label[predicted_label])