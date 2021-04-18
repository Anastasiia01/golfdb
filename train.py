from dataloader import HandwashDB, Normalize, ToTensor
from model import Handy
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os


if __name__ == '__main__':

    # training configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    split = 1
    iterations = 2000
    it_save = 100  # save model every 100 iterations
    n_cpu = 6
    seq_length = 64
    bs = 22  # batch size
    k = 10  # frozen layers

    model = Handy(pretrain=False, #change for True on GPU
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          device = device,
                          bidirectional=True,
                          dropout=False
                          )
    freeze_layers(k, model)
    model.train()
    model.to(device)

    dataset = HandwashDB(data_file='data/train_split_{}.pkl'.format(split),
                        vid_dir='data/handwash_videos_160/',
                        seq_length=seq_length,
                        transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                        train=True)
    print(dataset.__getitem__(20)) #checking that dataset is properly defined

    """data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = AverageMeter()

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    while i < iterations:
        for sample in data_loader:
            images, labels = sample['images'].cuda(), sample['labels'].cuda()
            logits = model(images)
            labels = labels.view(bs*seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/swingnet_{}.pth.tar'.format(i))
            if i == iterations:
                break
    """



