import pandas as pd
import os
import cv2
from multiprocessing import Pool
import numpy as np


df = pd.read_pickle('handwash.pkl')

video_dir = 'handwashing_videos/'
destination_path = 'handwash_videos_{}/'.format(160)


def preprocess_videos(video_name, dim=160):
    """
    Extracts relevant frames from videos
    """

    a = df.loc[df['video_name'] == video_name]
    events = a['events'].item() #if confident that a single row will be returned
    # events = a['events'].values[0] #select first row in an array of returned objects 
    # with one corresponding to a single row



    if not os.path.isfile(os.path.join(destination_path, "{}.mp4".format(video_name))):
        print('Processing video with name {}.mp4'.format(video_name))
        video_path = os.path.join(video_dir, '{}.mp4'.format(video_name))
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(os.path.join(destination_path, "{}.mp4".format(video_name)),
                              fourcc, cap.get(cv2.CAP_PROP_FPS), (dim, dim))
        """x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[0])
        y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[1])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[2])
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[3])"""
        count = 0
        success, frame = cap.read()

        # print(success)
        """frame = frame[:, :, [2, 1, 0]]
        image = Image.fromarray(frame)"""
        while success:
            count += 1
            # using frames from 3 sec before hands appear and 3 sec after they disappear.
            if count >= events[0]-90 and count <= events[-1]+90: 
                    """crop_img = image[y:y + h, x:x + w]
                    crop_size = crop_img.shape[:2]
                    ratio = dim / max(crop_size)
                    new_size = tuple([int(x*ratio) for x in crop_size])
                    resized = cv2.resize(crop_img, (new_size[1], new_size[0]))"""
                    resized = cv2.resize(frame, (dim, dim))
                    """delta_w = dim - new_size[1]
                    delta_h = dim - new_size[0]
                    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                    left, right = delta_w // 2, delta_w - (delta_w // 2)
                    b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                               value=[0.406*255, 0.456*255, 0.485*255]) """ # ImageNet means (BGR) as a color of border
                    out.write(resized)
            if count > events[-1]+90:
                break
            success, frame = cap.read()
    else:
        print('Video with name {} already completed for size {}'.format(video_name, dim))


if __name__ == '__main__':
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)
    #preprocess_videos(df.video_name[0]) # to preprocess a single video
    p = Pool(6) #multiprocessesing
    p.map(preprocess_videos, df.video_name) # to preprocess all videos
