#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import os
import time
import torch
import wandb
import shutil
import argparse

import numpy as np

from torch import nn
from tqdm import tqdm
from PIL import Image
from torch.optim import Adam
from datetime import datetime
from torch.utils.data import DataLoader
from codecarbon import OfflineEmissionsTracker
from sklearn.metrics import roc_auc_score, roc_curve

from models.msam import MSAM
from data.visa import ViSADataset
from data.mvtec import MVTecDataset
from utils.pt_to_onnx import PtToOnnx
from models.kairos_ad import KairosAD


def get_class_weights(dataset):
    labels = [x[1] for x in dataset]

    num_pos_class = sum(labels)
    num_neg_class = len(labels) - num_pos_class

    return (num_pos_class / len(labels), num_neg_class / len(labels))


def save_random_images(test_dataset, outs_raw, dataset_name,
                       category, save_dir, number_of_images=4):
    sigmoid_out = torch.nn.functional.sigmoid(torch.tensor(outs_raw,
                                                           dtype=torch.float32))
    prediction_labels = (sigmoid_out > 0.5).int().numpy()

    test_labels = np.array([label for _,label in test_dataset])
    fpr, tpr, thresholds = roc_curve(test_labels, outs_raw)
    youden_index = tpr - fpr
    best_threshold = thresholds[youden_index.argmax()]

    prediction_labels = (outs_raw > best_threshold).astype(int)

    classification_results = (prediction_labels == test_labels)

    true_indices = np.where(classification_results)[0]
    false_indices = np.where(~classification_results)[0]

    selected_true = np.random.choice(true_indices, min(number_of_images, len(true_indices)))
    selected_false = np.random.choice(false_indices, min(number_of_images, len(false_indices)))

    save_path = f'{save_dir}/qualitative/{category}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for index in np.concat([selected_true ,selected_false]):
        image, label = test_dataset[index]
        image = Image.fromarray(image)
        image.save(f'{save_path}/image_{time.time_ns()}_{label}_{sigmoid_out[index]}.jpeg')

    print(f'Succesfully saved images in {save_path}')


def test(classifier, test_loader, loss_function, device):

    classifier.eval()
    running_test_loss = 0

    labels = np.array([])
    outs_raw = np.array([])

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
            for test_images, test_labels in test_loader:
                test_images = test_images.to(device)
                test_labels = test_labels.to(device, torch.float)

                out_raw = classifier(test_images).view(-1)
                loss = loss_function(out_raw, test_labels)

                outs_raw = np.concatenate([outs_raw, out_raw.detach().cpu().numpy()])
                labels = np.concatenate([labels, test_labels.detach().cpu().numpy()])
                running_test_loss += loss.item() * test_images.size(0)
                pbar.set_postfix({'loss': running_test_loss / len(test_loader.dataset)})
                pbar.update(1)

    avg_val_loss = running_test_loss / len(test_loader.dataset)
    auroc = roc_auc_score(labels, outs_raw)

    return  avg_val_loss, auroc, outs_raw


def train(num_epochs, classifier, train_loader, test_loader, optim,
          loss_function, device, early_stoping, out_dir,
          category, calculate_emission=False):

    best_outs_raw = None
    emision_tracker = None
    best_auroc = -1*np.inf

    # Calculate training emission.
    if calculate_emission:
        emision_tracker = OfflineEmissionsTracker(project_name=f'{category}_train_emissions_data',
                                                  country_iso_code='ITA',
                                                  output_dir=out_dir,
                                                  log_level='critical',
                                                  allow_multiple_runs=True)
        emision_tracker.start()

    for epoch in range(num_epochs):
        classifier.train()

        running_train_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for images, labels in train_loader:
                optim.zero_grad()
                images = images.to(device)
                labels = labels.to(device, torch.float)

                out_raw = classifier(images).view(-1)
                loss = loss_function(out_raw, labels)

                loss.backward()
                optim.step()

                running_train_loss += loss.item() * images.size(0)
                pbar.set_postfix({'loss': running_train_loss / len(train_loader.dataset)})
                pbar.update(1)

        avg_train_loss = running_train_loss / len(train_loader.dataset)

        if calculate_emission:
            print(f'Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.5f}')
        else:
            avg_val_loss, auroc, outs_raw = test(classifier, test_loader, loss_function, device)

            if auroc > best_auroc:
                best_auroc = auroc
                best_outs_raw = outs_raw

            print(f'Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.5f} | Test Loss: {avg_val_loss:.5f} | AUROC: {auroc:.5f} | Best AUROC: {best_auroc:.5f}')

            if early_stoping and auroc >= 1.0:
                return best_auroc, best_outs_raw

    if calculate_emission:
        emision_tracker.stop()
        emision_tracker = OfflineEmissionsTracker(project_name=f'{category}_train_emissions_data',
                                                  country_iso_code='ITA',
                                                  output_dir=out_dir,
                                                  log_level='critical',
                                                  allow_multiple_runs=True)

        # Calculate testing emission.
        emision_tracker.start()
        avg_val_loss, auroc, outs_raw = test(classifier, test_loader, loss_function, device)
        print(f'Test Loss: {avg_val_loss:.5f} | AUROC: {auroc:.5f}')
        emision_tracker.stop()

        return auroc, outs_raw

    return best_auroc, best_outs_raw


def main(args):
    print(f'Args: {args}')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Set seed for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')

    dt_string = datetime.now().strftime('%d-%m-%Y-%H-%M')
    run_name = f'{args.dataset_name}_C-{args.category_names}_#L-{args.num_of_layers}_LD-{args.layer_divider}_BS-{args.batch_size}_E-{args.num_epochs}_{dt_string}'

    run_dir = os.path.join(args.save_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    wandb_run = None
    if args.use_wandb:
        wandb.init(
            project=args.wandb_projects,
            name=run_name,
            entity=args.wandb.entity,
            config=args)
        wandb_run = wandb.run

    model_vit = 'vit_t'
    sam_checkpoint = 'MobileSAM/weights/mobile_sam.pt'

    all_categories = None
    if args.dataset_name == 'mvtec':
        all_categories = MVTecDataset.categories
    elif args.dataset_name == 'visa':
        all_categories = ViSADataset.categories

    training_categories = all_categories if 'all' in args.category_names else args.category_names

    results = dict()
    for category in training_categories:
        print('++++++++++++++++++++++++++++++')
        print(f'Start training and testing on: {category}')
        print('++++++++++++++++++++++++++++++')

        msam = MSAM(model_vit, sam_checkpoint, device)
        classifier = KairosAD(msam, args.num_of_layers, args.layer_divider).to(device)

        # Calculate the model parameters.
        print(f'KairosAD no. of parameters: {sum(p.numel() for p in classifier.parameters() if p.requires_grad)}')

        if args.pretrained:
            classifier.load_state_dict(torch.load('Weights/MSAM_AD.pt', weights_only=True))

        optim = Adam(classifier.parameters(), lr=args.learning_rate)

        train_loader = None
        test_loader = None

        if args.dataset_name == 'mvtec':
            dataset = MVTecDataset(args.dataset_path, category=category)
            dataset_test = dataset.get_dataset(False)
            train_loader = DataLoader(dataset, args.batch_size, shuffle=True)
            test_loader = DataLoader(dataset_test, args.batch_size, shuffle=False)
        elif args.dataset_name == 'visa':
            dataset_train = ViSADataset(args.dataset_path, category=category,
                                        type='train', shot='highshot')
            dataset_test = ViSADataset(args.dataset_path, category=category,
                                       type='test', shot='highshot')
            train_loader = DataLoader(dataset_train, args.batch_size,
                                      shuffle=True, drop_last=True)
            test_loader = DataLoader(dataset_test, args.batch_size,
                                     shuffle=False)

        pos_weight = None
        if args.dataset_name == 'visa':
            neg_weight, pos_weight = get_class_weights(dataset_train)
            print(f'Negative weights: {neg_weight:.5f} | Positive weights: {pos_weight:.5f}')
            pos_weight = torch.tensor(pos_weight * 5, dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        auroc, outs_raw = train(args.num_epochs, classifier,
                                train_loader, test_loader, optim,
                                criterion, device, args.early_stoping,
                                run_dir, category, calculate_emission=args.calculate_emission)

        if args.qualitative_results:
            save_random_images(dataset_test, outs_raw, args.dataset_name, 
                               category, run_dir, 4)
        results[category] = auroc

    print(f'Summary:\n {results}')
    if wandb_run:
        wandb_run.log({'class_results' : wandb.Table(columns=['Class', 'Value'], 
                                                     data=[[k, v] for k, v in results.items()])})

    # Save the model results.
    with open(os.path.join(run_dir, f'quantitative.txt'), 'w') as f:
        for k, v in results.items():
            f.write(f'{k}: {v}\n')

        averege_auroc = sum(results.values()) / len(results)
        f.write(f'Averege AUROC: {averege_auroc}\n')
        if wandb_run:
            wandb_run.log({'averege_auroc': averege_auroc})

    # Save the model weights.
    torch.save(classifier.state_dict(), os.path.join(run_dir, f'KairosAD_weights.pt'))

    # Export the model to ONNX format.
    classifier.eval()
    exporter = PtToOnnx(classifier)
    exporter.export(os.path.join(run_dir, f'KairosAD_ONNX_weights.pt'))

    if wandb_run:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General parameters.
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--ckpt_path', type=str, default='./weights')

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_projects', type=str, default='')

    # Model parameters.
    parser.add_argument('--num_of_layers', type=int, default=5)
    parser.add_argument('--layer_divider', type=int, default=2)

    # Training specific parameters.
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # Dataset specific parameters.
    parser.add_argument('--dataset_name', type=str, choices=['mvtec', 'visa'], required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--category_names', type=str, choices=[*MVTecDataset.categories, *ViSADataset.categories ,'all'], nargs='+', required=True)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--export_onnx', action='store_true')
    parser.add_argument('--early_stoping', action='store_true')
    parser.add_argument('--calculate_emission', action='store_true')
    parser.add_argument('--qualitative_results', action='store_true')

    args = parser.parse_args()
    main(args)
