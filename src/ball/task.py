import os
import torch
import argparse
import shutil
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from src.ball.model import BallTrackerNet
from src.ball.dataloader import GetLoader, BallDataset
from ball.utils import postprocessing

class Task:
    def __init__(self, config, device, dataloader):
        self.train_loader, self.val_loader = dataloader
        self.device = device
        self.max_iters = config["train"]["max_iters"]
        self.checkpoint = config["train"]["checkpoint_path"]
        self.num_epochs = config["train"]["num_epochs"]
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, model, optimizer, epoch):
        losses = []
        train_progress_bar = tqdm(self.train_loader, colour="cyan")
        for iter, batch in enumerate(train_progress_bar):
            model.train()
            images, gts = batch
            images, gts = images.to(device=self.device), gts.to(device=self.device)
            output = model(images)
            loss = self.criterion(output, gts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            train_progress_bar.set_description('Train: Epoch: {}/{}. Iter: {}/{}. Loss: {}'.format(epoch+1, self.num_epochs, iter+1, self.max_iters, loss.item()))

            if iter > self.max_iters - 1:
                break
        return np.mean(losses)
    
    def val(self, model, epoch):
        losses = []
        model.eval()

        val_progress_bar = tqdm(self.val_loader, colour="yellow")
        for iter, batch in enumerate(val_progress_bar):
            with torch.no_grad():
                images, gts = batch
                images, gts = images.to(device=self.device), gts.to(device=self.device)
                output = model(images)
                loss = self.criterion(output, gts)
                losses.append(loss)
                val_progress_bar.set_description('Val: Epoch: {}/{}. Iter: {}/{}. Loss: {}'.format(epoch+1, self.num_epochs, iter+1, len(self.val_loader), loss.item()))
        return np.mean(losses)






            

        