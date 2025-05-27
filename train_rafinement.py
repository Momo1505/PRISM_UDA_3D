from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import os

import matplotlib.pyplot as plt

from torch import optim
from mmseg.models.uda.refinement import EncodeDecode
from tqdm import tqdm
from Datasets import RefinementDataset,GTADataset
from refinement_module import UNet
import argparse
from mmseg.models.uda.swinir_backbone import MGDNRefinement



def compute_iou(preds:torch.Tensor,gt:torch.Tensor,num_classes=19,ignore_index=255):
    mask = gt != ignore_index
    preds = preds[mask]
    gt = gt[mask]

    iou_list = []

    for cls in range(num_classes):
        cls_in_preds = preds == cls
        cls_in_gt = gt == cls

        intersection = (cls_in_preds & cls_in_gt).sum().item()
        union = (cls_in_preds | cls_in_gt).sum().item()

        if union == 0:
            iou_list.append(0)
            continue  # this class is not present in gt

        iou = intersection / union
        iou_list.append(iou)

    if not iou_list:
        raise ValueError("No class found in this predictions")
    return sum(iou_list) / len(iou_list)

def plot(pl:torch.Tensor,
         sam:torch.Tensor,
         preds:torch.Tensor,
         gt:torch.Tensor,
         epoch,
         batch_id,
         model_name,
         dataset_type,mask_type
         ):
    if batch_id % 100==0:
        pl = pl.squeeze().detach().cpu().numpy()
        sam = sam.squeeze().detach().cpu().numpy()
        preds = preds.squeeze().detach().cpu().numpy()
        gt = gt.squeeze().detach().cpu().numpy()

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

        axes[0].imshow(pl)
        axes[0].set_title("Pseudo label de l'EMA")

        axes[1].imshow(sam)
        axes[1].set_title("SAM")

        axes[2].imshow(preds)
        axes[2].set_title("Prédictions du raffineur")

        axes[3].imshow(gt)
        axes[3].set_title("Ground truth")

        plt.tight_layout()

        os.makedirs(f"refinement_results/{model_name}_train_on_{dataset_type}_mask_type_{mask_type}/debug", exist_ok=True)
        plt.savefig(f"refinement_results/{model_name}_train_on_{dataset_type}_mask_type_{mask_type}/debug/{epoch}_{batch_id}.png")
        plt.close(fig)

def save_metrics(train_losses,val_losses,train_m_iou,val_m_iou,model_name,dataset_type,mask_type):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_m_iou, label="Train mIoU")
    plt.plot(val_m_iou, label="Val mIoU")
    plt.legend()
    plt.title("Mean IoU")
    plt.savefig(f"refinement_results/{model_name}_train_on_{dataset_type}_mask_type_{mask_type}/metrics.png")
    plt.close()


def one_batch(model,model_name,epoch,device,
              batch,batch_id,
              optimizer:optim.Optimizer,
              loss_function:nn.CrossEntropyLoss,
              dataset_type:str,mask_type,
              on_training=True):
    model.train() if on_training else model.eval()
    if dataset_type == "cityscapes":
        pl,sam,gt = batch
        pl,sam,gt = pl.float(),sam.float(),gt.long()
        pl,sam,gt = pl.to(device),sam.to(device),gt.to(device)
        preds = model(sam,pl,apply_mask=on_training) if model_name == "Unet" else model(sam,pl)
    else :
        sam,gt = batch
        sam,gt = sam.float(),gt.long()
        sam,gt = sam.to(device),gt.to(device)
        preds = model(sam,sam,apply_mask=on_training) if model_name == "Unet" else model(sam,sam)
    loss = loss_function(preds,gt.squeeze(1))
    if on_training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preds = preds.argmax(dim=1)

    iou = compute_iou(preds,gt.squeeze(1))

    if not on_training :
        if dataset_type == "cityscapes":
            plot(pl,sam,preds,gt,epoch,batch_id,model_name,dataset_type,mask_type)
        else :
            plot(sam,sam,preds,gt,epoch,batch_id,model_name,dataset_type,mask_type)

    return loss.item(), iou


def one_epoch(model,model_name,epoch,device,
              dataloader: DataLoader,
              optimizer:optim.Optimizer,
              loss_function:nn.CrossEntropyLoss,
              dataset_type:str,mask_type,
              on_training=True):
    train_loss = 0
    m_iou = 0
    N = 0
    for batch_id,batch in enumerate(dataloader):
        loss,iou = one_batch(model=model,model_name=model_name,mask_type=mask_type,epoch=epoch,device=device,batch=batch,batch_id=batch_id,optimizer=optimizer,dataset_type=dataset_type, loss_function=loss_function,on_training=on_training)
        train_loss = train_loss + loss
        m_iou = m_iou + iou
        N += 1
    return train_loss / N, m_iou / N

def train(model,max_epochs,lr,train_dl,val_dl,model_name,dataset_type,mask_type):
    train_losses = []
    val_losses = []
    train_m_iou = []
    val_m_iou = []
    device = "cuda" #if torch.cuda.is_available() else "cpu"
    best_val_iou = 1000

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(),lr=lr)
    loss_function = nn.CrossEntropyLoss(ignore_index=255)
    progress_bar = tqdm(range(max_epochs))

    if dataset_type == "cityscapes":
        train_dataset = "cityscapes"
        val_dataset = "cityscapes"
    else:
        train_dataset = "gta"
        val_dataset = "gta"
    for epoch in progress_bar:
        train_loss, train_iou = one_epoch(model=model,model_name=model_name,mask_type=mask_type,epoch=epoch,device=device,dataloader=train_dl,optimizer=optimizer,dataset_type=train_dataset,loss_function=loss_function,on_training=True)
        train_losses.append(train_loss)
        train_m_iou.append(train_iou)
        with torch.no_grad():
            val_loss, val_iou = one_epoch(model=model,model_name=model_name,mask_type=mask_type,epoch=epoch,device=device,dataloader=val_dl,optimizer=optimizer,dataset_type=val_dataset,loss_function=loss_function,on_training=False)
            val_losses.append(val_loss)
            val_m_iou.append(val_iou)
        progress_bar.set_postfix(dict(train_loss=train_loss,train_iou=train_iou,val_loss=val_loss,val_iou=val_iou))
        # save losses and IoU
        save_metrics(train_losses,val_losses,train_m_iou,val_m_iou,model_name,dataset_type,mask_type=mask_type)
        if val_iou < best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou
            }, f"refinement_results/{model_name}_train_on_{dataset_type}_mask_type_{mask_type}/best_checkpoint.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name',choices=['Unet',"MGDN"] ,type=str)

    parser.add_argument(
        '--dataset-type',choices=['cityscapes',"gta"] ,type=str)
    
    parser.add_argument(
        '--mask-type',choices=['binary',"colour"] ,type=str)

    args = parser.parse_args()

    model_name = args.model_name
    dataset_type = args.dataset_type
    mask_type = args.mask_type

    if dataset_type == "cityscapes":
        if mask_type == "colour":
            train_ds = RefinementDataset(mode="train")
            val_ds = RefinementDataset(mode="val")
            train_dl = DataLoader(train_ds,8,True,num_workers=8)
            val_dl = DataLoader(val_ds,1,False,num_workers=8)
        else:
            train_ds = RefinementDataset(mode="train",mask_type=mask_type)
            val_ds = RefinementDataset(mode="val",mask_type=mask_type)
            train_dl = DataLoader(train_ds,8,True,num_workers=8)
            val_dl = DataLoader(val_ds,1,False,num_workers=8)
    elif dataset_type == "gta":
        if mask_type == "colour":
            train_ds = GTADataset(mode="train")
            val_ds = GTADataset(mode="val")

            train_dl = DataLoader(train_ds,8,True,num_workers=8)
            val_dl = DataLoader(val_ds,1,False,num_workers=8)
        else :
            train_ds = GTADataset(mode="train",mask_type=mask_type)
            val_ds = GTADataset(mode="val",mask_type=mask_type)

            train_dl = DataLoader(train_ds,8,True,num_workers=8)
            val_dl = DataLoader(val_ds,1,False,num_workers=8)

    if model_name == "Unet":
        in_channel = 2 if dataset_type == "cityscapes" else 1
        model = UNet(in_channel=in_channel,n_classes=19,dataset_type=dataset_type)
    elif model_name == "MGDN":
        model = MGDNRefinement()
    

    print(f"training {model_name} on {dataset_type} mask type {mask_type}, len du training is {len(train_ds)}")

    train(model,100,1e-4,train_dl,val_dl,model_name,dataset_type,mask_type)