from model import Handy
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import HandwashDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds


def eval(model, split, seq_length, n_cpu, device, disp):
    dataset = HandwashDB(data_file='data/val_split_{}.pkl'.format(split),
                        vid_dir='data/handwash_videos_160/',
                        seq_length=seq_length,
                        transform=transforms.Compose([ToTensor(),
                                                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                        train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        # images.shape is [1, frames_count_per_video, 3, 160, 160]
        while batch * seq_length < images.shape[1]: # images.shape[1] is frames_count_per_video
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        _, _, _, _, c = correct_preds(probs, labels.squeeze())
        if disp:
            print(i, c)
        correct.append(c)
    PCE = np.mean(correct) # Percentage of Correct Events
    return PCE


if __name__ == '__main__':

    split = 1
    seq_length = 64
    n_cpu = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Handy(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          device = device,
                          bidirectional=True,
                          dropout=False)

    save_dict = torch.load('models/net_1800.pth.tar')
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    PCE = eval(model, split, seq_length, n_cpu, device, True)
    print('Average PCE: {}'.format(PCE))


