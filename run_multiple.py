
from main import main

datasets = ["sst2"]#["sst2small", "mrpcsmall", "mnlismall", "qnlismall","sst2","mrpc" ,"cola", "qnli","mnli"]#[ ]#,"mnli"]
split_by = ["layer", "qkv"]#"layer","qkv",
n_opts = [1,4,7,10]
models = ["bert"]#, "roberta"]
update_rule = ["cycle"] #, "impact_mag"]#"cycle", 
optim = ["adamsls"]#, "sgd", "sgdsls"]"adam", 
combine = [0]


def create_config(name, ds, split, n_opt, model, opt, update_r = "cycle", i = 0, combine = 0):
    with open(name, 'w') as f:
            f.write("[DEFAULT]\n")
            f.write("batch_size = 32\n")
            f.write("checkpoint = None\n")
            f.write("directory = results/"  + ds + opt + str(n_opt) + model + split + update_r + str(i) + "\n")
            f.write("seed = 42\n")
            f.write("epochs = 5\n")
            f.write("dataset = " + ds + "\n")
            f.write("optim = " + opt + "\n")
            f.write("num_diff_opt =" + str(n_opt) + "\n")
            f.write("model = " + model + "\n")
            f.write("split_by = " + split + "\n")
            f.write("update_rule = " + update_r + "\n")
            f.write("combine = " + str(0) + "\n")
   # print("results/"  +ds + opt+ str(n_opt) + model + split )
    main(name)

for ds in datasets:
    for model in models:
        for opt in optim:
            if "sls" in opt:
                for update_r in update_rule:
                    for split in split_by:
                        if split == "layer":
                            for n_opt in n_opts:
                                for comb in combine:
                                    for i in range(5):
                                        create_config("config_gen.json", ds, split, n_opt, model , opt, update_r, i,combine = comb)
                        else:
                            for i in range(5):
                                create_config("config_gen.json", ds, split, 1, model , opt, update_r, i)
            else:
                for i in range(5):
                    create_config("config_gen.json", ds, "layer", 1, model , opt,"cycle", i)


            
            
            
            
            
            
            
           
            
            


