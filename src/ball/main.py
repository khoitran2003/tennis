import os
import tqdm
import yaml
import torch
import shutil
import argparse
from typing import Text
from torch.utils.tensorboard import SummaryWriter

from src.ball.task import Task
from src.ball.dataloader import GetLoader
from src.ball.model import BallTrackerNet


def get_args():
    parsers = argparse.ArgumentParser("ball")
    parsers.add_argument(
        "--config_path",
        type=str,
        default="cfg/ball.yaml",
    )
    parsers.add_argument("--mode", type=str, default="train")
    args = parsers.parse_args()
    return args


def main(args) -> None:

    with open(args.config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    dataloader = GetLoader(batch_size=config["batch_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    task = Task(config=config, device=device, dataloader=dataloader.load_train_val())

    if args.mode == "train":

        model = BallTrackerNet(out_channels=config["train"]["out_channels"])
        optimizer = torch.optim.Adadelta(
            params=model.parameters(), lr=config["train"]["lr"]
        )

        if os.path.isdir(
            os.path.join(config["train"]["checkpoint_path"], "ball_last.pt")
        ):
            checkpoint = torch.load(
                os.path.join(config["train"]["checkpoint_path"], "ball_last.pt")
            )
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_loss = checkpoint["best_loss"]
        else:
            start_epoch = 0
            best_loss = 10000

        if os.path.isdir(config["train"]["log_path"]):
            shutil.rmtree(config["train"]["log_path"])
        else:
            os.makedirs(config["train"]["log_path"])

        writer = SummaryWriter(config["train"]["log_path"])
        print("Starting training...")
        for epoch in range(start_epoch, config["train"]["num_epochs"]):
            loss = task.train(model=model, optimizer=optimizer, epoch=epoch)
            writer.add_scalar("Train/Loss", loss, epoch)
            if (epoch > 0) & (epoch % config["val"]["val_intervals"] == 0):
                task.val(model=model, epoch=epoch)
                loss = writer.add_scalar("Train/Loss", loss, epoch)
                writer.add_scalar("Val/Loss", loss, epoch)
                checkpoint = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                }
                torch.save(
                    checkpoint,
                    os.path.join(config["train"]["checkpoint_path"], "ball_last.pt"),
                )
                if loss < best_loss:
                    best_loss = loss
                    torch.save(
                        checkpoint,
                        os.path.join(
                            config["train"]["checkpoint_path"], "ball_best.pt"
                        ),
                    )
        print("Training finish")


if __name__ == "__main__":
    args = get_args()
    main(args)
