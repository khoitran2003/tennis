import torch.nn as nn
import torch
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import ast
import os 

from src.ball.model import BallTrackerNet
from src.ball.utils import (
    load_model,
    read_vid,
    std2birdeyeview,
    ball_infer,
    remove_outliers,
    split_track,
    interpolation,
)


def get_args():
    parser = argparse.ArgumentParser(description="Infer")
    parser.add_argument(
        "--ball_checkpoint", type=str, default="checkpoints/ball_best.pt"
    )
    parser.add_argument(
        "--court_checkpoint", type=str, default="checkpoints/court_best.pt"
    )
    parser.add_argument("--video_input", type=str, default="test_video.mp4")
    return parser.parse_args()


def main(args):
    id_output = 0
    if not os.path.isdir("results"):
        os.makedirs("results")
        os.makedirs("results/output1")

    else:
        for entry in os.scandir("results"):
            if entry.is_dir():
                id_output +=1
        os.makedirs(f"results/output{id_output+1}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames, fps, w_f, h_f = read_vid(args.video_input)

    # ball tracking
    ball_model = BallTrackerNet(out_channels=256).to(device=device)
    ball_model = load_model(
        ball_model, checkpoint_path=args.ball_checkpoint, device=device
    )
    ball_model.eval()

    print("BALL INFERENCE...")
    ball_track, dists = ball_infer(
        frames, ball_model, device=device, height=360, width=640
    )
    ball_track = remove_outliers(ball_track, dists)
    subtracks = split_track(ball_track)
    for r in subtracks:
        ball_subtrack = ball_track[r[0] : r[1]]
        ball_subtrack = interpolation(ball_subtrack)
        ball_track[r[0] : r[1]] = ball_subtrack

    balls = []
    for point in ball_track:
        if point[0] is not None and point[1] is not None:
            point = [int(point[0]), int(point[1])]
        else:
            point = [0, 0]
        balls.append(point)

    # players tracking
    print("PLAYERS INFERENCE...")
    yolo_model = YOLO("checkpoints/yolov8n.pt")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    std_path = f"results/output{id_output+1}/std_view.mp4"  
    bev_path = f"results/output{id_output+1}/bev_view.mp4"  
    frame_rate = fps  
    std_size = (w_f, h_f)  
    bev_size = (360, 640)

    std_writer = cv2.VideoWriter(std_path, fourcc, frame_rate, std_size)
    bev_writer = cv2.VideoWriter(bev_path, fourcc, frame_rate, bev_size)

    for id in range(len(frames)):
        results = yolo_model.track(frames[id], persist=True)
        annotated_frame = results[0].plot()
        if results[0].boxes.xyxy is not None and results[0].boxes.id is not None:
            bc = [
                [int(box[0]), int(box[1] + box[3] / 2)]
                for box in results[0].boxes.xywh.cpu()
            ]
            ids = results[0].boxes.id.int().cpu().tolist()
            # mini map
            bev_frame = std2birdeyeview(
                ball_track=balls[id],
                players=bc,
                color_court=(255, 0, 255),
                color_player=(0, 0, 255),
                size=(640, 360),
            )

            # track ball, player
            for i in range(7):
                if id - i > 0:
                    if ball_track[id - i][0]:
                        x = int(ball_track[id - i][0])
                        y = int(ball_track[id - i][1])
                        cv2.circle(
                            annotated_frame,
                            (x, y),
                            radius=0,
                            color=(0, 0, 255),
                            thickness=10 - i,
                        )
                    else:
                        break
            std_writer.write(annotated_frame)
            bev_writer.write(bev_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_args()
    main(args)
