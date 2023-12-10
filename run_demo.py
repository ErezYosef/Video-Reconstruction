import argparse
import os
import torch

import numpy as np
import torchvision

import cv2
from concurrent.futures import ThreadPoolExecutor
from ablation_nets import UNet_generic

class real_imgs_ds_fi(torch.utils.data.Dataset):
    def __init__(self, Nsamples = 7, imgpath=''):

        self.Nsamples = Nsamples
        #self.crf_inv = self._init_inv_crf()
        # Paths:
        self.imgpath = imgpath
        #print(self.imglist)
        #print(len(self))
        self.transform = None
        self.t_data = (torch.linspace(0, 2, Nsamples) - 1).to(dtype=torch.float32)
        x = cv2.imread(self.imgpath, cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        depth = 2 ** 16 - 1 if x.dtype == np.uint16 else 2 ** 8 - 1
        x = torch.from_numpy(x / depth).permute(2, 0, 1)
        if self.transform is not None:
            x = self.transform(x)
        self.input = x

    def __getitem__(self, frame_t):
        t = self.t_data[frame_t].view(1)
        return self.input, t, frame_t

    def __len__(self):
        return self.Nsamples


def save_thread(imgs_pred, newpath, frame_idxs, crfunc=None):
    for frm in range(len(imgs_pred)):
        # tens = torch.stack([gtimg[i], outimg[i]])
        # print(tens.shape)
        # tens = torchvision.utils.make_grid(crf(tens), nrow=2)
        if crfunc is None:
            crfunc = lambda x: x
        torchvision.utils.save_image(crfunc(imgs_pred[frm]),
                                     os.path.join(newpath, f'f{frame_idxs[frm]:02}.png'))

def real_visualize_fi(net, device, args):
    num_workers = args.num_workers
    bs = args.batch_size
    imgpath = args.imgpath

    test_ds = real_imgs_ds_fi(Nsamples=args.time_samples, imgpath=imgpath)
    #val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    #print(len(val_ds), len(tst_ds), len(val_ds.dataset), len(tst_ds.dataset))

    print('visualize the model')
    newpath = args.imgpath[:-4].replace('inputs', 'outputs')
    if not os.path.isdir(newpath):
        os.makedirs(newpath)

    net.eval()
    #coded_img_lst = []
    for i, batch in enumerate(loader):
        # print(i, end = '\r')
        imgs, t_data, frame_idxs = batch
        #img_name = test_ds.imglist[i][:-4]
        imgs = imgs.to(device=device, dtype=torch.float32)
        t_data = t_data.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            imgs_pred = net(imgs, t_data)

        save_thread(imgs_pred, newpath, frame_idxs, crfunc=lambda x: x**0.5)

# python run_demo.py --lf final_weights.pt  -t 3 --imgpath img1_demo.png
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--load', dest='load', type=str, default='',
                        help='Load model path for dir of .pth file')
    parser.add_argument('--lf', dest='load_file', type=str, default=None,
                        help='Name of the .pt file to load')
    parser.add_argument('--workers', dest='num_workers', type=int, default=0,
                        help='Num_workers for dataloader')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=2,
                        help='loader batch size')
    parser.add_argument('-t', '--time_samples', dest='time_samples', type=int, default=25,
                        help='Number of frames to generate for an image')

    parser.add_argument('--imgpath', dest='imgpath', type=str, default='inputs/car.png',
                        help='image path for a real image')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet_generic(bilinear=True,  posenc=5, time_concat=True)

    net.to(device=device)
    filename = args.load_file
    if filename is None:
        filename = 'final_weights.pt'
        args.load_file = filename
    checkpoint = torch.load(os.path.join(args.load, filename), map_location=device)

    net.load_state_dict(checkpoint['model'])

    print(args)
    try:
        real_visualize_fi(net=net, device=device, args=args)
    except KeyboardInterrupt:
        print('Test Interrupted. exit.')