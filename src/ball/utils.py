import torch
from tqdm import tqdm
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
import cv2
from itertools import groupby
from scipy.spatial import distance


def creat_gaussian(s, variance=10):
    x, y = np.mgrid[-s : s + 1, -s : s + 1]
    g = np.exp(-(x**2 + y**2) / float(2 * variance))
    gaussian_kernel_array = (g * 255 / g[int(len(g) / 2)][int(len(g) / 2)]).astype(
        np.uint8
    )
    return gaussian_kernel_array


def fill_gt_on_hm(heatmap, temp, x, y):
    start_hm_x = 0
    start_hm_y = 0
    end_hm_x = None
    end_hm_y = None
    start_temp_x = 0
    start_temp_y = 0
    end_temp_x = None
    end_temp_y = None
    if 20 <= x <= 1260 and 20 <= y <= 700:
        heatmap[y - 20 : y + 21, x - 20 : x + 21] = temp
    else:
        if x < 20:
            end_hm_x = x + 21
            start_temp_x = abs(x - 20)
        if x > 1260:
            start_hm_x = x - 21
            end_temp_x = abs(1280 - x) + 21
        if y < 20:
            end_hm_y = y + 21
            start_temp_y = abs(y - 20)
        if y > 700:
            start_hm_y = y - 21
            end_temp_y = abs(720 - y) + 21
        if 20 <= x <= 1260:
            start_hm_x = x - 20
            end_hm_x = x + 21
        if 20 <= y <= 700:
            start_hm_y = y - 20
            end_hm_y = y + 21

        heatmap[start_hm_y:end_hm_y, start_hm_x:end_hm_x] = temp[
            start_temp_y:end_temp_y, start_temp_x:end_temp_x
        ]
    return heatmap


def transform(frame, width=640, height=360):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transfomred_frame = Compose(
        [ToPILImage(), Resize(size=(height, width)), ToTensor()]
    )
    return transfomred_frame(frame)


def hm2xy(feature_map, scale=2):
    feature_map *= 255
    feature_map = feature_map.reshape((360, 640))
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=2,
        minRadius=2,
        maxRadius=7,
    )
    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0] * scale
            y = circles[0][0][1] * scale
    return x, y


def read_vid(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, width_frame, height_frame


def load_model(model, checkpoint_path, device):
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model
    except Exception as e:
        return f"Error: {e}"


def postprocess(output_map, scale=2):
    output_map *= 255
    output_map = output_map.reshape((360, 640))
    output_map = output_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(output_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=2,
        minRadius=2,
        maxRadius=7,
    )
    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0] * scale
            y = circles[0][0][1] * scale
    return x, y


def ball_infer(frames, model, device, height, width):
    """Run pretrained model on a consecutive list of frames
    :params
        frames: list of consecutive video frames
        model: pretrained model
    :return
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """

    dists = [-1] * 2
    ball_track = [(None, None)] * 2
    for num in tqdm(range(1, len(frames) - 1)):
        img = cv2.resize(frames[num - 1], (width, height))
        img_prev = cv2.resize(frames[num], (width, height))
        img_preprev = cv2.resize(frames[num - 1], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)
    return ball_track, dists


def remove_outliers(ball_track, dists, max_dist=100):
    """Remove outliers from model prediction
    :params
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
        max_dist: maximum distance between two neighbouring ball points
    :return
        ball_track: list of ball points
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i + 1] > max_dist) | (dists[i + 1] == -1):
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i - 1] == -1:
            ball_track[i - 1] = (None, None)
    return ball_track


def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    """Split ball track into several subtracks in each of which we will perform
    ball interpolation.
    :params
        ball_track: list of detected ball points
        max_gap: maximun number of coherent None values for interpolation
        max_dist_gap: maximum distance at which neighboring points remain in one subtrack
        min_track: minimum number of frames in each subtrack
    :return
        result: list of subtrack indexes
    """
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if (k == 1) & (i > 0) & (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor - 1], ball_track[cursor + l])
            if (l >= max_gap) | (dist / l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1
        cursor += l
    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])
    return result


def interpolation(coords):
    """Run ball interpolation in one subtrack
    :params
        coords: list of ball coordinates of one subtrack
    :return
        track: list of interpolated ball coordinates of one subtrack
    """

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])

    track = [*zip(x, y)]
    return track


def frame_track(frame, ball_track, path_output_video, fps, trace=7):
    """Write .avi file with detected ball tracks
    :params
        frames: list of original video frames
        ball_track: list of ball coordinates
        path_output_video: path to output video
        fps: frames per second
        trace: number of frames with detected trace
    """
    height, width = frame[0].shape[:2]
    out = cv2.VideoWriter(
        path_output_video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    for num in range(len(frame)):
        frame = frame[num]
        for i in range(trace):
            if num - i > 0:
                if ball_track[num - i][0]:
                    x = int(ball_track[num - i][0])
                    y = int(ball_track[num - i][1])
                    frame = cv2.circle(
                        frame, (x, y), radius=0, color=(0, 0, 255), thickness=10 - i
                    )
                else:
                    break
        out.write(frame)
    out.release()


def draw(
    frame,
    color_court: tuple[int],
    color_player: tuple[int],
    thickness: int,
    points: list,
    players: list,
):
    cv2.line(
        frame,
        (int(points[0][0]), int(points[0][1])),
        (int(points[1][0]), int(points[1][1])),
        color_court,
        thickness,
    )
    cv2.line(
        frame,
        (int(points[1][0]), int(points[1][1])),
        (int(points[2][0]), int(points[2][1])),
        color_court,
        thickness,
    )
    cv2.line(
        frame,
        (int(points[2][0]), int(points[2][1])),
        (int(points[3][0]), int(points[3][1])),
        color_court,
        thickness,
    )
    cv2.line(
        frame,
        (int(points[3][0]), int(points[3][1])),
        (int(points[0][0]), int(points[0][1])),
        color_court,
        thickness,
    )

    cv2.line(
        frame,
        (int(points[4][0]), int(points[4][1])),
        (int(points[5][0]), int(points[5][1])),
        color_court,
        thickness,
    )
    cv2.line(
        frame,
        (int(points[6][0]), int(points[6][1])),
        (int(points[7][0]), int(points[7][1])),
        color_court,
        thickness,
    )

    cv2.line(
        frame,
        (int(points[8][0]), int(points[8][1])),
        (int(points[9][0]), int(points[9][1])),
        color_court,
        thickness,
    )
    cv2.line(
        frame,
        (int(points[10][0]), int(points[10][1])),
        (int(points[11][0]), int(points[11][1])),
        color_court,
        thickness,
    )

    cv2.line(
        frame,
        (int(points[12][0]), int(points[12][1])),
        (int(points[13][0]), int(points[13][1])),
        color_court,
        thickness,
    )
    for player in players:
        cv2.circle(frame, (int(player[0]), int(player[1])), 5, color_player, -1)

    cv2.circle(frame, (int(points[14][0]), int(points[14][1])), 3, (255, 255, 255), -1)
    return frame


def std2birdeyeview(
    ball_track: list,
    color_court,
    color_player,
    players: list,
    size: tuple[int],
):
    tl = (428 - 50, 162 - 50)
    bl = (0, 720)
    tr = (850 + 50, 162 - 50)
    br = (1280, 720)

    matrix = cv2.getPerspectiveTransform(
        np.float32([tl, bl, tr, br]),
        np.float32([(0, 0), (0, size[0]), (size[1], 0), (size[1], size[0])]),
    )

    point_in_original_frame = np.array(
        [
            [222, 567],
            [1063, 567],
            [850, 162],
            [428, 162],
            [330, 567],
            [480, 162],
            [957, 567],
            [798, 162],
            [386, 416],
            [897, 416],
            [819, 218],
            [462, 218],
            [642, 416],
            [642, 218],  # p2
            ball_track,  # ball
        ],
        dtype=np.float32,
    )
    transformed_point = cv2.perspectiveTransform(
        point_in_original_frame[None, :, :], matrix
    )

    player_in_original_frame = np.array(players, dtype=np.float32)

    transformed_players_point = cv2.perspectiveTransform(
        player_in_original_frame[None, :, :], matrix
    )

    zero_frame = np.zeros(shape=(size[0], size[1], 3), dtype=np.uint8)
    trans_frame = draw(
        frame=zero_frame,
        color_court=color_court,
        color_player=color_player,
        thickness=2,
        points=transformed_point[0],
        players=transformed_players_point[0],
    )
    return trans_frame
