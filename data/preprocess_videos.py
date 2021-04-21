import pandas as pd
import os
import cv2
from multiprocessing import Pool
from threading import Lock
import numpy as np
from statistics import mean


class Preprocessing(object):
    """Preprocesses videos by extracts relevant frames from thwm"""
    def __init__(self, video_dimensions = 160):
        self.df = pd.read_pickle('handwash.pkl')
        self.dim = video_dimensions
        self.input_dir = 'handwashing_videos/'
        self.output_dir = 'handwash_videos_{}/'.format(self.dim)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.ratioArr = []
        self.framesCountArr = []
        #self.lock = Lock()
        self.frames_till_events = 20

    def preprocess_videos(self, video_name):
        """
        Extracts relevant frames from videos
        """

        a = self.df.loc[self.df['video_name'] == video_name]
        events = a['events'].item() #if confident that a single row will be returned
        # events = a['events'].values[0] #select first row in an array of returned objects 
        # with one corresponding to a single row

        if not os.path.isfile(os.path.join(self.output_dir, "{}.mp4".format(video_name))):
            print('Processing video with name {}.mp4'.format(video_name))
            original_video_path = os.path.join(self.input_dir, '{}.mp4'.format(video_name))
            cap = cv2.VideoCapture(original_video_path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(os.path.join(self.output_dir, "{}.mp4".format(video_name)),
                                fourcc, cap.get(cv2.CAP_PROP_FPS), (self.dim, self.dim))
            full_count = 0
            success, frame = cap.read()
            num_frames = 0  
            start_frame = None
            # print(success)
            """frame = frame[:, :, [2, 1, 0]]
            image = Image.fromarray(frame)"""
            while success:
                # using frames from 3 sec before hands appear and 3 sec after they disappear.
                if full_count%3==0:
                    count = full_count//3 # actual rate is 30fps, we need 10fps
                    if count >= events[0]-self.frames_till_events and count <= events[-1]+self.frames_till_events: 
                        if (start_frame==None):
                            start_frame = count
                        resized = cv2.resize(frame, (self.dim, self.dim))
                        out.write(resized)
                    if count > events[-1]+self.frames_till_events:
                        count = count - 1
                        break
                full_count += 1
                success, frame = cap.read()

            end_frame = count
            #print(f"start: {start_frame} and end: {end_frame}")
            total_used_frames = end_frame - start_frame + 1
            ratioNoneventVsEvent = (total_used_frames - 4)/4
            self.ratioArr.append(ratioNoneventVsEvent)
            self.framesCountArr.append(total_used_frames)

            #print(f"total number of frames is {total_used_frames}")
            #print(f"Then ratioEventVsNonevent is {ratioNoneventVsEvent}")

        else:
            print(f'Video with name {video_name} already completed for size {self.dim}')


if __name__ == '__main__':
    prep = Preprocessing()
    #prep.preprocess_videos(prep.df.video_name[3]) # to preprocess a single video
    #pool = Pool(6) #multiprocessesing
    #pool.map(prep.preprocess_videos, prep.df.video_name) # to preprocess all videos

    for idx in range(len(prep.df.video_name)):
        #print(idx)
        prep.preprocess_videos(prep.df.video_name[idx]) 
    print(f"Arr \n {prep.ratioArr}")
    print(f"Mean is {mean(prep.ratioArr)}" )
    print(f" Number of ratios is {len(prep.ratioArr)}")
    print(f"Frames Arr \n {prep.framesCountArr}")
    print(f"Avg frame number is {mean(prep.framesCountArr)}" )
    print(f" Number of frame counts is {len(prep.framesCountArr)}")
