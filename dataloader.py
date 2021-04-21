import os.path as osp
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class HandwashDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        events = a['events']
        frame_count = a['total_frames'].item()

        max_frames_till_events = 20
        start = max(0, events[0]-max_frames_till_events)
        events -= start  # now frames correspond to frames in preprocessed video clips
        end = min(frame_count-1, events[-1]+max_frames_till_events )        
        
        images, labels = [], []
        cap = cv2.VideoCapture(osp.join(self.vid_dir, '{}.mp4'.format(a['video_name'])))

        if self.train:
            # random starting position, sample 'seq_length' frames

            """choose_event = np.random.randint(0, 4)            
            interval_center = events[choose_event]
            interval_start = max(0, events[choose_event] - 100)
            interval_end = min(end, events[choose_event] + 100)
            #print(f"Selected event is {choose_event} and its frame is {interval_center}")

            start_frame = np.random.randint(interval_start, interval_end + 1)"""

            start_frame = np.random.randint(start, end+1)
            #print(f"Start frame {start_frame}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) #set position
            pos = start_frame
            
            while len(images) < self.seq_length:
                ret, frame = cap.read()
                if ret:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    if pos in events:
                        labels.append(np.where(events == pos)[0][0]) #np.where(events == pos) returns the index where condition is true
                    else:
                        labels.append(4)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
            cap.release()
        else:
            # full clip
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, frame = cap.read()
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(img)
                if pos in events:
                    labels.append(np.where(events == pos)[0][0]) # any of action classes
                else:
                    labels.append(4) # label no-class
            cap.release()

        sample = {'images':np.asarray(images), 'labels':np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}


if __name__ == '__main__':

    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std (RGB)

    dataset = HandwashDB(data_file='data/train_split_1.pkl',
                        vid_dir='data/handwash_videos_160/',
                        seq_length=64,
                        transform=transforms.Compose([ToTensor(), norm]),
                        train=False)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 4)[0]
        print('{} events: {}'.format(len(events), events))




    





       

