import itertools
import os
import argparse
from pathlib import Path
import wandb
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import WeightedRandomSampler, DataLoader
#from torchsummary import summary
from torchvision.transforms import transforms
from model.train import BreastDensityClassifier
from dataloader.data_loader import MammographyDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from utils.utils_functions import set_seed

from multiprocessing import freeze_support


# os.environ['WANDB_SILENT']="true"

def main(args):
    read_excel = pd.read_csv(args.configuration_file)

    for index, row in read_excel.iterrows():
        print('Index', index)
        wandb.login()
        run = wandb.init(project=args.project_name, reinit=True, dir = args.experiment_dir)
        wandb_logger = pl.loggers.WandbLogger()
        with run:
            try:
                torch.cuda.empty_cache()
                #wandb_logger = pl.loggers.WandbLogger()
                config = wandb.config
                config.cv_run = row['no']
                set_seed(config.cv_run)
                config.experiment = row['experiment']
                #set_seed(config.cv_run) #askk
                #config.dataset = row['dataset']
                config.task = row['task']
                config.model = row['model']
                config.view_position = row['view_position']
                #config.method = row['method']
                config.pretrained = row['pretrained']
                config.loss = row['loss']
                config.use_sampler = row['use_sampler']
                config.batch_size = row['batch_size']
                config.image_height = row['image_height']
                config.image_width = row['image_width']
                config.num_worker = row['num_worker']
                #config.optimizer = row['optimizer']
                config.learning_rate = row['lr']
                #config.weight_decay = row['weight_decay']
                #config.momentum = row['sgd_momentum']
                #config.drop_out = row['drop_out']
                config.num_epochs = row['epochs']
                config.patience = row['patience']
                #config.temperature = row['temp_scl']
                config.out_folder = row['out_folder']
                #if args.images_folder:
                #    data_folder = args.images_folder
                #else:
                data_folder = row['data_folder']
                #print(data_folder)
                #keyword = row['data_folder_keyword'] #ask
                excel_file_train = row['train']
                excel_file_validation = row['valid']
                excel_file_test = row['test']

                if config.task == 'BreastDensity':
                    breast_density_classes = ['not dense', 'dense']
                elif config.task == 'BreastDensity4':
                    breast_density_classes = ['DENSITY A', 'DENSITY B', 'DENSITY C', 'DENSITY D']

                #os.makedirs(config.out_folder, exist_ok=True)
                #wandb.init(dir=config.out_folder)

                num_classes = len(breast_density_classes)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print('Device : ', device)
                #weight, samples_weight = compute_sample_weights(excel_file_train, class_type=config.dataset)
                train_transforms = transforms.Compose([
                    # transforms.ToPILImage(),
                    # transforms.Resize(256),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),#check this
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # check this
                ])

                valid_transforms = transforms.Compose([
                    # transforms.ToPILImage(),
                    # transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    # test should have the normalization as well
                ])

                test_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                train_dataset = MammographyDataset(split_file=excel_file_train,
                                                           #config.dataset,
                                                           data_folder=data_folder,
                                                           view_position=config.view_position,
                                                           task=config.task,
                                                           #keyword,
                                                           image_height=config.image_height,
                                                           image_width=config.image_width,
                                                           transform=train_transforms)
                valid_dataset = MammographyDataset(split_file=excel_file_validation,
                                                           #config.dataset,
                                                           data_folder=data_folder,
                                                           #keyword,
                                                           view_position=config.view_position,
                                                           task=config.task,
                                                           image_height=config.image_height,
                                                           image_width=config.image_width,
                                                           transform=valid_transforms)
                test_dataset = MammographyDataset(split_file=excel_file_test,
                                                   #config.dataset,
                                                            data_folder=data_folder,
                                                            view_position=config.view_position,
                                                            task=config.task,
                                                            #keyword,
                                                            image_height=config.image_height,
                                                            image_width=config.image_width,
                                                            transform=test_transforms)

                if config.use_sampler:
                    samples_weights = train_dataset.get_weights()
                    train_sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)
                    shuffle = False
                else:
                    train_sampler = None
                    shuffle = True

                train_loader = DataLoader(train_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=shuffle,sampler=train_sampler,
                                          num_workers=config.num_worker,
                                          pin_memory=True)
                valid_loader = DataLoader(valid_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          num_workers=config.num_worker,
                                          pin_memory=True)
                test_loader = DataLoader(test_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         num_workers=config.num_worker,
                                         pin_memory=True)
                model = BreastDensityClassifier(targets=breast_density_classes,num_classes=num_classes,
                                                model_type=config.model,
                                                loss_type=config.loss,
                                                pretrained=config.pretrained,
                                                learning_rate = config.learning_rate)
                checkpoint_callback = ModelCheckpoint(
                    monitor='val_loss',
                    filename='{epoch}-{step}-{val_loss:.2f}',
                    mode='min',
                    save_top_k=1,
                    save_last=False
                )
                #checkpoint_callback = CustomModelCheckpoint(
                #    run_no = config.cv_run,
                #    model_name = config.model,
                #    view_position = config.view_position,              
                #    monitor='val_loss',
                #    filename='{epoch}-{step}-{run_no}-{model_name}-{val_loss:.2f}-{view_position}',
                #    mode='min',
                #    save_top_k=1,
                #    save_last=False
                #)
                trainer = pl.Trainer(max_epochs=config.num_epochs,
                                    accelerator='gpu', devices=1,
                                    default_root_dir=config.out_folder, 
                                    logger=wandb_logger,
                                    callbacks=[checkpoint_callback,
                                    EarlyStopping(monitor="val_loss", mode="min",patience=config.patience)]
                                    )
                trainer.fit(model, train_loader, valid_loader)
                trainer.test(dataloaders=test_loader)
               
            except KeyboardInterrupt:
                print('Closing Loop:  Jumping to the next experiment')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('  Mammography', add_help=False)
    parser.add_argument("--configuration_file", type=Path, default='/cluster/eq27ifuw/Breast_Mammo/experiment_code/ex5/config.csv',
                        help="""Change hyperparameters and input the filepath of CSV file""")
    parser.add_argument("--project_name", type=str, default='Breast_Mammo',
                        help="""Project name for wandb""")
    parser.add_argument("--experiment_dir", type=Path, default='/cluster/eq27ifuw/Breast_Mammo/experiment_code/ex5/',
                        help="""Add the directory of experiment""")
    args = parser.parse_args()
    freeze_support()
    main(args)