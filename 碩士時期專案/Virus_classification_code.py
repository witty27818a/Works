import os
import argparse
import numpy as np
from PIL import Image
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from resnest.torch.models import resnest50, resnest101 好像不是這樣寫
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from tqdm import tqdm

class VirusModel(nn.Module):
    def __init__(self, name, mode, classnum = 6, pretrained = True):
    # def __init__(self, name, mode, classnum = 24, pretrained = True):
        super().__init__()
        self.mode = mode
        if name == "resnest50":
            # self.model = resnest50(pretrained = pretrained)
            self.model = torch.hub.load("zhanghang1989/ResNeSt", "resnest50", pretrained = pretrained)
        elif name == "resnest101":
            # self.model = resnest101(pretrained = pretrained)
            self.model = torch.hub.load("zhanghang1989/ResNeSt", "resnest101", pretrained = pretrained)
        else:
            raise NameError("Unknown Resnest Model Name!")
        
        if pretrained:
            for param in self.model.parameters():
                param.required_grads = False

        # change last linear layer.
        fc_in_num = self.model.fc.in_features # 2048? 要再確定是不是+確定真的換對layer
        if self.mode == "single":
            self.model.fc = nn.Linear(fc_in_num, classnum, bias = True)
        else:
            self.model.fc = nn.Sequential(
                nn.Linear(fc_in_num, classnum, bias = True),
                nn.Sigmoid()
            )

    def forward(self, X):
        outputs = self.model(X)

        return outputs

class VirusDataset(Dataset):
    def __init__(self, root, mode, mean = (0.5, 0, 0.5), std = (0.5, 0.01, 0.5)):
        '''
        normalize的部分，R=255*256*(x-0.5)^4*(x+0.5)^4，G=0，B=255x^2。x是entropy
        If x = 0~0.5，R=G=B=0，else(x > 0.5)，就照上面公式
        '''
        self.path = os.path.join(root, mode)
        # 此外，我們需要一個檔名和label一一對應的csv檔案
        "label請使用one-hot-vectors"
        self.img_list = os.listdir(self.path)
        self.mapping = {"Adware": 0, "Backdoor": 1, "Ransom": 2, "Trojan": 3, "Worm": 4, "Normal": 5}
        self.transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.img_list[idx]))
        img = self.transformations(img)
        label = torch.zeros(6)
        for key, value in self.mapping.items():
            if self.img_list[idx].find(key) != -1:
                label[value] = 1.0
        return img, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help = "resnest50 or resnest101", type = str, default = "resnest50")
    parser.add_argument("--model_path", help = "the path to store model weigths", type = str, default = "Final_Project")
    parser.add_argument("--root", help = "the path of the data root directory", type = str, default = "data") # 下層資料夾要有single/multi，在下面要有train/test之類
    parser.add_argument("--mode", help = "single or multi", type = str, default = "single")
    parser.add_argument("--mean", help = "mean for normalization", nargs = "+", type = float, default = [0.5, 0, 0.5])
    parser.add_argument("--std", help = "std for normalization", nargs = "+", type = float, default = [0.5, 0.01, 0.5])
    parser.add_argument("--lr", help = "learning rate", type = float, default = 1e-3)
    # parser.add_argument("--gamma", help = "Multiply gamma to learning rate until milestone", type = float, default = 1.0)
    # parser.add_argument("--milestone", help = "How many epochs we multiply gamma", type = int, default = 10)
    parser.add_argument("--batch", help = "batch size", type = int, default = 32)
    parser.add_argument("--epochs_FE", help = "epochs for feature extraction", type = int, default = 2)
    parser.add_argument("--epochs_FT", help = "epochs for fine-tuning", type = int, default = 18)
    parser.add_argument("--test_only", help = "0 for train+test, 1 for test only", type = int, default = 0)
    parser.add_argument("--gpu", help = "To run on which GPU, a number", type = int, default = 0)
    args = parser.parse_args()

    lr = args.lr
    # gamma = args.gamma
    # milestone = args.milestone
    batch_size = args.batch
    epochs_FE = args.epochs_FE # Feature extraction
    epochs_FT = args.epochs_FT # Fine-tuning
    mean = tuple(args.mean)
    std = tuple(args.std)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    model_path = args.model_path
    os.makedirs(model_path, exist_ok = True)

    train_dataset = VirusDataset(args.root, os.path.join(args.mode, "train"), mean, std)
    test_dataset = VirusDataset(args.root, os.path.join(args.mode, "test"), mean, std)
    train_dataloader = DataLoader(train_dataset, batch_size, True)
    test_dataloader = DataLoader(test_dataset, batch_size, False)
    print("# of train batches: {}".format(len(train_dataloader)))
    print("# of test batches: {}".format(len(test_dataloader)))

    if args.mode == "single":
        model = VirusModel(args.model, args.mode)
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == "multi":
        model = VirusModel(args.model, args.mode)
        loss_fn = nn.BCELoss()
    else:
        raise NameError("Not Supported Mode!")
    model.to(device)

    if not args.test_only:
        with open(os.path.join(model_path, "settings.txt"), "w") as settings:
            settings.write(args.model + "\n")
            settings.write(args.mode + "\n")
            settings.write("mean " + str(mean) + "\n")
            settings.write("std " + str(std) + "\n")
            settings.write("lr " + str(lr) + "\n")
            # settings.write("gamma " + str(gamma) + "\n")
            # settings.write("milestone " + str(milestone) + "\n")
            settings.write("batch_size " + str(batch_size) + "\n")
            settings.write("epochs_FE " + str(epochs_FE) + "\n")
            settings.write("epochs_FT " + str(epochs_FT) + "\n")

        train_losses = []
        best_loss = float("inf")
        progress = tqdm(total = epochs_FE + epochs_FT)

        print("start training......")
        model.train()
        params_FC_layer = []
        for param in model.parameters():
            if param.requires_grad:
                params_FC_layer.append(param)
        optimizer = optim.Adam(params_FC_layer, lr = lr) # 看要不要AdamW，還有要不要調參?
        # scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor = gamma, total_iters = milestone)
        for e in range(epochs_FE):
            losses = 0
            for imgs, labels in train_dataloader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                if args.mode == "single":
                    labels = torch.max(labels, 1)[1]

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                losses += loss.item()

                loss.backward()
                optimizer.step()
            
            epoch_loss = losses / len(train_dataloader)
            train_losses.append(epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(deepcopy(model.state_dict()), os.path.join(model_path, "best_model_" + args.mode + ".pt"))
            
            # scheduler.step()
            progress.update(1)
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr = lr)
        # scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor = gamma, total_iters = max(milestone - epochs_FE, 0))
        for e in range(epochs_FT):
            losses = 0
            for imgs, labels in train_dataloader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                if args.mode == "single":
                    labels = torch.max(labels, 1)[1]

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                losses += loss.item()

                loss.backward()
                optimizer.step()
            
            epoch_loss = losses / len(train_dataloader)
            train_losses.append(epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(deepcopy(model.state_dict()), os.path.join(model_path, "best_model_" + args.mode + ".pt"))
            
            # scheduler.step()
            progress.update(1)
        
        np.save(os.path.join(model_path, "train_losses_" + args.mode + ".npy"), np.array(train_losses))
        progress.close()

    # test
    print("start testing......")
    predictions = torch.Tensor()
    ground_truths = torch.Tensor()
    model.load_state_dict(torch.load(os.path.join(model_path, "best_model_" + args.mode + ".pt"), device))
    model.eval()

    with torch.no_grad():
        for imgs, labels in test_dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            if args.mode == "single":
                # outputs = F.one_hot(torch.max(outputs, 1)[1], num_classes = 6)
                outputs = torch.max(outputs, 1)[1]
                labels = torch.max(labels, 1)[1]
            else:
                outputs = (outputs > 0.5)
            predictions = torch.cat((predictions, outputs.cpu()), dim = 0)
            ground_truths = torch.cat((ground_truths, labels), dim = 0)
        if args.mode == "single":
            cm = confusion_matrix(ground_truths.numpy(), predictions.numpy())
        else:
            cm = multilabel_confusion_matrix(ground_truths.numpy(), predictions.numpy())
    
    np.save(os.path.join(model_path, "confusion_matrixes_" + args.mode + ".npy"), cm)
    print("Confusion Matrix:")
    print(cm) 
    
'''
對於多標籤(multi-label)分類問題，我用BCE搭配模型最後一層接sigmoid喔
然後使用multilabel_confusion_matrix
也可以參考一下LAB5的evaluator.py的寫法? 或是其他寫法
'''

'''
其他resnest使用的tricks
1. label smoothing: 本來one-hot-encoding是[0,0,1,0]之類的，label smoothing就是透過一個factor把機率分給其他0
1的機率變成1-factor，0的機率變成(factor/# of categories)，所以本來機率是[0,0,1,0]，如果factor=0.1，就變成[0.33,0.33,0.9,0.33]

2. auto augmentation: 太複雜沒看

3. large mini-batch distributed training: 基本一般人做不到

4. mixup-training

5. large crop-size
'''