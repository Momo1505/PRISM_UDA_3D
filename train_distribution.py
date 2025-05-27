from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import os
from torch import tensor
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR

from torch import optim
from tqdm import tqdm
from Datasets import DistributionDataset
from models import BinaryMaskTransformer
import argparse
import glob

def collate_fn(data):
    masks,true_labels = zip(*data)
    return torch.cat(masks),torch.cat(true_labels)

def save_metrics(train_losses,val_losses,train_m_iou,val_m_iou,model_name,dataset_type):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_m_iou, label="Train acc")
    plt.plot(val_m_iou, label="Val acc")
    plt.legend()
    plt.title("Mean acc")
    os.makedirs(f"refinement_results/{model_name}_train_on_{dataset_type}/debug", exist_ok=True)
    plt.savefig(f"refinement_results/{model_name}_train_on_{dataset_type}/metrics.png")
    plt.close()


def one_batch(model,model_name,epoch,device,
              batch,batch_id,
              optimizer:optim.Optimizer,scheduler,
              loss_function:nn.CrossEntropyLoss,
              dataset_type:str,
              on_training=True):
    model.train() if on_training else model.eval()
    x,y = batch
    x,y = x.float().to(device), y.long().to(device)

    preds = model(x)
    loss = loss_function(preds,y)
    
    if on_training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_id % 100 == 0: 
            print("train_loss = ",loss.item()," epoch = ",epoch," batch_id = ",batch_id)
    pred = preds.argmax(dim=-1)
    correct = (pred == y).sum().item()
    total = y.size(0)
    accuracy = correct / total

    return loss.item(),accuracy
    

def one_epoch(model,model_name,epoch,device,
              dataloader: DataLoader,
              optimizer:optim.Optimizer,
              scheduler,
              loss_function:nn.CrossEntropyLoss,
              dataset_type:str,
              on_training=True):
    train_loss = 0
    acc = 0
    N = 0
    for batch_id,batch in enumerate(dataloader):
        loss,accuracy = one_batch(model=model,model_name=model_name,epoch=epoch,device=device,batch=batch,batch_id=batch_id,optimizer=optimizer,dataset_type=dataset_type, loss_function=loss_function,on_training=on_training,scheduler=scheduler)
        train_loss = train_loss + loss
        acc = acc + accuracy
        N += 1
    return train_loss / N, acc / N

def train(model,max_epochs,lr,train_dl,val_dl,model_name,dataset_type):
    train_losses = []
    val_losses = []
    train_acc = [] 
    val_acc = []
    device = "cuda" #if torch.cuda.is_available() else "cpu"
    best_acc = 1000

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    steps_per_epoch = len(train_dl)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,  # peak learning rate
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # % of cycle spent increasing LR
        anneal_strategy='cos',  # or 'linear'
        div_factor=25.0,  # initial LR = max_lr / div_factor
        final_div_factor=1e4,  # final LR = initial_lr / final_div_factor
    )

    loss_function = nn.CrossEntropyLoss(ignore_index=255)
    progress_bar = tqdm(range(max_epochs))

    if dataset_type == "cityscapes":
        train_dataset = "cityscapes"
        val_dataset = "cityscapes"
    else:
        train_dataset = "gta"
        val_dataset = "gta"
    for epoch in progress_bar:
        train_loss, train_accuracy = one_epoch(model=model,model_name=model_name,scheduler=scheduler,epoch=epoch,device=device,dataloader=train_dl,optimizer=optimizer,dataset_type=train_dataset,loss_function=loss_function,on_training=True)
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)
        with torch.no_grad():
            val_loss, val_accuracy = one_epoch(model=model,model_name=model_name,scheduler=scheduler, epoch=epoch,device=device,dataloader=val_dl,optimizer=optimizer,dataset_type=val_dataset,loss_function=loss_function,on_training=False)
            val_losses.append(val_loss)
            val_acc.append(val_accuracy)
        progress_bar.set_postfix(dict(train_loss=train_loss,train_acc=train_accuracy,val_loss=val_loss,val_acc=val_accuracy))
        # save losses and IoU
        save_metrics(train_losses,val_losses,train_accuracy,val_accuracy,model_name,dataset_type)
        if val_accuracy < best_acc:
            best_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_accuracy
            }, f"refinement_results/{model_name}_train_on_{dataset_type}/best_checkpoint.pth")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name',choices=["transformer"] ,type=str,default="transformer")

    parser.add_argument(
        '--dataset-type',choices=["gta"] ,type=str,default="gta")
    
    parser.add_argument(
        '--num-layers',type=int,default=5)

    args = parser.parse_args()

    model_name = args.model_name
    dataset_type = args.dataset_type
    num_layers = args.num_layers


    if model_name == "transformer":
        model = BinaryMaskTransformer(num_layers=num_layers,patch_size=16)

    labels_list = glob.glob(os.path.join("data/gta","labels","*_labelTrainIds.png"),recursive=True)
    training_size =int( 0.9 * len(labels_list))
    train_label = labels_list[:training_size]
    val_label = labels_list[training_size:]

    train_ds = DistributionDataset(train_label)
    val_ds = DistributionDataset(val_label)

    train_dl = DataLoader(train_ds,batch_size=8,shuffle=True,num_workers=8,collate_fn=collate_fn)
    val_dl = DataLoader(val_ds,batch_size=5,shuffle=True,num_workers=8,collate_fn=collate_fn)

    print(f"training {model_name} on {dataset_type} num_layers {num_layers}, len du training is {len(train_ds)}")

    train(model,50,1e-4,train_dl,val_dl,model_name,dataset_type)