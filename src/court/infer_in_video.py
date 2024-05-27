import os
import cv2
import numpy as np
import torch
from src.court.tracknet import CourtTrackerNet
import torch.nn.functional as F
from tqdm import tqdm
from src.court.postprocess import postprocess, refine_kps
from src.court.homography import get_trans_matrix, refer_kps
import argparse

def read_video(path_video):
    """ Read video file
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

def write_video(imgs_new, fps, path_output_video):
    height, width = imgs_new[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'),
                          fps, (width, height))
    for num in range(len(imgs_new)):
        frame = imgs_new[num]
        out.write(frame)
    out.release() 

def court_infer(frames, model, device, height, width):
    use_refine_kps = True
    use_homography = True
    frames_upd = []
    for image in tqdm(frames):
        img = cv2.resize(image, (width, height))
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        out = model(inp.float().to(device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
            if use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
            points.append((x_pred, y_pred))

        if use_homography:
            matrix_trans = get_trans_matrix(points)
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                points = [np.squeeze(x) for x in points]

        for j in range(len(points)):
            if points[j][0] is not None:
                image = cv2.circle(image, (int(points[j][0]), int(points[j][1])),
                                radius=0, color=(0, 0, 255), thickness=10)
        frames_upd.append(image)
    return frames_upd




    
    