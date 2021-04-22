import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from eval import ToTensor, Normalize
from model import Handy
import numpy as np
import torch.nn.functional as F
from PIL import Image
from data.preprocess_videos import prep

# Version that correctly distinguishes wrist appear and wrist disappear events.

event_names = {
    0: 'Wrist Appears',
    1: 'Start Handwashing',
    2: 'End Handwashing',
    3: 'Wrist Disappears',
}


class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx): 
        cap = cv2.VideoCapture(self.path)

        # preprocess and return frames   
        images = []
        count = 0 
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, frame = cap.read()
            if count%3==0:            
                resized = cv2.resize(frame, (self.input_size, self.input_size))
                img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                images.append(img)
            count+=1            
        cap.release()
        labels = np.zeros(len(images)) # only for compatibility with transforms
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_probs(images, model, device, seq_length):
  batch = 0
  while batch * seq_length < images.shape[1]:
      if (batch + 1) * seq_length > images.shape[1]:
          image_batch = images[:, batch * seq_length:, :, :, :]
      else:
          image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
      logits = model(image_batch.to(device))
      if batch == 0:
          probs = F.softmax(logits.data, dim=1).cpu().numpy()
      else:
          probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
      batch += 1
  return probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video that you want to test', default='test_video.mp4')
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=128)
    args = parser.parse_args()
    seq_length = args.seq_length

    print('Preparing video: {}'.format(args.path))

    ds = SampleVideo(args.path, transform=transforms.Compose([ToTensor(),
                                Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])]))

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = Handy(pretrain=True,
                        width_mult=1.,
                        lstm_layers=1,
                        lstm_hidden=256,
                        device = device,
                        bidirectional=True,
                        dropout=False)

    try:
        save_dict = torch.load('models/net_v2_1600.pth.tar')
    except:
        print("Model weights not found. Download model weights and place in 'models' folder. See README for instructions")

    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")

    print('Testing...')
    for sample in dl: 
        images = sample['images']
        #print(images.shape) torch.Size([1, 414, 3, 160, 160])
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        probs = get_probs(images, model, device, seq_length)
    
    prep_images = prep.all_frames
    images = []
    for i, frame in enumerate(prep_images):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(img)
        #image = frame[:, :, [2, 1, 0]]
        #image = Image.fromarray(image)
        #image.save(f'/content/drive/MyDrive/preprocessed_video_frames3/{i}.jpg')
    labels = np.zeros(len(images)) # only for compatibility with transforms
    sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
    transform = transforms.Compose([ToTensor(),
                                Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])])
    sample = transform(sample)        
    images = sample['images']
    images = images.view(1, images.shape[0], images.shape[1], images.shape[2], images.shape[3]) 
    probs = get_probs(images, model, device, seq_length)

    events = np.argmax(probs, axis=0)[:-1]
    print('Predicted event frames: {}'.format(events))   
    cap = cv2.VideoCapture(args.path)

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Condifence: {}'.format([np.round(c, 3) for c in confidence]))

    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        _, frame = cap.read()
        frame = frame[:, :, [2, 1, 0]]
        image = Image.fromarray(frame)
        image.save(f'/content/drive/MyDrive/handwashing_results/{event_names[i]}-{confidence[i]}.jpg')


