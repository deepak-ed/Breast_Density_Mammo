import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import wandb
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
import torchvision.transforms as transforms
import pandas as pd
from dataloader.data_loader import MammographyDataset
from torchvision import models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
import torch.nn.functional as F
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from pytorch_lightning.loggers import WandbLogger


# Define the custom PyTorch Lightning module
class BreastDensityClassifier(pl.LightningModule):
    def __init__(self,targets,num_classes,model_type, loss_type,pretrained,learning_rate):
        super(BreastDensityClassifier, self).__init__()
        self.targets = targets
        self.num_classes = num_classes
        self.lr = learning_rate
        if pretrained:
            self.model = torch.hub.load("pytorch/vision", model_type, weights="DEFAULT")
        else:
            self.model = torch.hub.load("pytorch/vision", model_type, weights=None)
        #if model_type == "resnet18":
        #    self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        if loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        print(labels.size())
        
        images_np = images.detach().cpu().numpy()

        # Log the images and labels for the first batch of the first epoch
        #if self.current_epoch == 0 and batch_idx == 0:
            #wandb.log({"Images": [wandb.Image(np.transpose(img, (1, 2, 0)), caption=str(label.item())) for img, label in zip(images_np, labels)]})
        #    for image, caption in zip(images_np, labels):
        #        img = np.transpose(image, (1, 2, 0))
        #        self.logger.experiment.log_image(key="Images", images=wandb.Image(img), caption=caption)
        outputs = self(images)
        print(outputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        print(preds)
        accuracy = torch.sum(preds == labels).item() / len(labels)
        
        # Calculate F1 score
        f1 = f1_score(labels.cpu(), preds.cpu(), average='macro')
        
        # Log accuracy, F1 score, AUC, and loss to WandB
        #wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy, "train_f1_score": f1})
        self.log('train_loss', loss.item(), rank_zero_only=True)
        self.log('train_accuracy',accuracy, rank_zero_only=True)
        self.log('train_f1_score', f1, rank_zero_only=True)
        #self.log('train_auc', auc, rank_zero_only=True)
        class_counts = torch.bincount(labels, minlength=self.num_classes).float()
        print(class_counts)
        # Normalize counts to get percentage
        class_percentage = class_counts / len(labels)

        # Log class-wise percentage distribution to WandB
        class_names = self.targets
        for i, class_name in enumerate(class_names):
            self.log(f"train_class_percentage_{class_name}", class_percentage[i].item())
        return {
            'loss': loss,
            'labels': labels,
            'preds': preds
                }

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(preds == labels).item() / len(labels)
        
        # Calculate F1 score
        f1 = f1_score(labels.cpu(), preds.cpu(), average='macro')     
        # Log accuracy, F1 score, AUC, and loss to WandB
        #wandb.log({'val_loss': loss.item(), 'val_accuracy': accuracy, 'val_f1_score': f1})
        self.log('val_loss',loss.item(), rank_zero_only=True)
        self.log('val_accuracy',accuracy, rank_zero_only=True)
        self.log('val_f1_score', f1, rank_zero_only=True)
        return {
            'loss': loss,
            'valid_labels': labels,
            'valid_preds': preds
                }
        
    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(preds == labels).item() / len(labels)
        f1 = f1_score(labels.cpu(), preds.cpu(), average='macro')
        #if self.num_classes == 2:
        #    auc = roc_auc_score(labels.cpu(), preds.cpu())

        self.log('test_loss', loss.item(), rank_zero_only=True)
        self.log('test_accuracy', accuracy, rank_zero_only=True)
        self.log('test_f1_score', f1, rank_zero_only=True)
        return {
            'loss': loss,
            'test_labels': labels,
            'test_preds': preds
                }

    def training_epoch_end(self, outputs):
        labels = torch.cat([x['labels'] for x in outputs], dim=0)
        preds = torch.cat([x['preds'] for x in outputs], dim=0)

        # Compute AUC-ROC score for training
        train_auc_roc_score = roc_auc_score(labels.cpu(), preds.cpu())
        self.log('train_auc_roc_score', train_auc_roc_score, on_epoch=True)

        # Compute confusion matrix and classification report
        train_conf_matrix = confusion_matrix(labels.cpu(), preds.cpu())
        train_class_report = classification_report(labels.cpu(), preds.cpu())

        # Log confusion matrix and classification report


    def validation_epoch_end(self, outputs):
        labels = torch.cat([x['valid_labels'] for x in outputs], dim=0)
        preds = torch.cat([x['valid_preds'] for x in outputs], dim=0)

        # Compute AUC-ROC score for validation
        val_auc_roc_score = roc_auc_score(labels.cpu(), preds.cpu())
        self.log('val_auc_roc_score', val_auc_roc_score, on_epoch=True)

        # Compute confusion matrix and classification report
        val_conf_matrix = confusion_matrix(labels.cpu(), preds.cpu())
        val_class_report = classification_report(labels.cpu(), preds.cpu())

        # Log confusion matrix and classification report
    def test_epoch_end(self, outputs):
        labels = torch.cat([x['test_labels'] for x in outputs], dim=0)
        preds = torch.cat([x['test_preds'] for x in outputs], dim=0)

        # Compute AUC-ROC score for testing
        test_auc_roc_score = roc_auc_score(labels.cpu(), preds.cpu())
        self.log('test_auc_roc_score', test_auc_roc_score, on_epoch=True)

        # Compute confusion matrix and classification report
        test_conf_matrix = confusion_matrix(labels.cpu(), preds.cpu())
        test_class_report = classification_report(labels.cpu(), preds.cpu())

        # Log confusion matrix and classification report

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)