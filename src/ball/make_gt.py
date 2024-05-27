import numpy as np
import argparse
import os
import shutil
import pandas as pd
import cv2
from ball.utils import creat_gaussian, fill_gt_on_hm


def get_args():
    parser = argparse.ArgumentParser("Make tennis ball data")
    parser.add_argument("--input", "-r", type=str, default="ball_data")
    parser.add_argument("--output", "-c", type=str, default="data/ball")
    return parser.parse_args()


def main(args):
    # check clf dir exist
    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    os.makedirs(os.path.join(args.output, "train"))
    os.makedirs(os.path.join(args.output, "val"))
    os.makedirs(os.path.join(args.output, "train", "images"))
    os.makedirs(os.path.join(args.output, "val", "images"))
    os.makedirs(os.path.join(args.output, "train", "labels"))
    os.makedirs(os.path.join(args.output, "val", "labels"))

    count = 0
    for game in os.listdir(args.input)[:-1]:
        for clip in os.listdir(os.path.join(args.input, game)):
            for file in os.listdir(os.path.join(args.input, game, clip)):
                if ".csv" in file:
                    csv_file = file
                    data = pd.read_csv(os.path.join(args.input, game, clip, csv_file))
                    data = data.rename(columns={"file name": "file_name"})
                    for image_file, vis, x, y in zip(
                        data["file_name"],
                        data["visibility"],
                        data["x-coordinate"],
                        data["y-coordinate"],
                    ):
                        heatmap = np.zeros((h, w), dtype=np.uint8)
                        if vis != 0:
                            x = int(x)
                            y = int(y)
                            image = cv2.imread(
                                os.path.join(args.input, game, clip, image_file)
                            )
                            h, w, _ = image.shape
                            heatmap = np.zeros((h, w), dtype=np.uint8)
                            temp = creat_gaussian(20)
                            heatmap = fill_gt_on_hm(heatmap, temp, x, y)
                            print(count)

                            if count < 15000:
                                cv2.imwrite(
                                    os.path.join(
                                        args.output, "train", "images", f"{count+1}.jpg"
                                    ),
                                    image,
                                )
                                cv2.imwrite(
                                    os.path.join(
                                        args.output, "train", "labels", f"{count+1}.jpg"
                                    ),
                                    heatmap,
                                )
                            else:
                                cv2.imwrite(
                                    os.path.join(
                                        args.output, "val", "images", f"{count+1}.jpg"
                                    ),
                                    image,
                                )
                                cv2.imwrite(
                                    os.path.join(
                                        args.output, "val", "labels", f"{count+1}.jpg"
                                    ),
                                    heatmap,
                                )
                            count += 1


if __name__ == "__main__":
    args = get_args()
    main(args)
    # test_case(args)
