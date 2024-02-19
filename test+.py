import torch
import torch.nn as nn
import logging
import sys
import os
import model_LANet
import numpy as np
from options.test_options import TestOptions
import natsort
from scipy import io
import imageio

def test_net(net,device):
    DATA_SIZE = opt.data_size
    test_results = os.path.join(opt.saveroot, 'test_results_V2+')
    net.eval()
    test_images = np.zeros((1, opt.channels,DATA_SIZE[1], DATA_SIZE[2]))
    testids = opt.test_ids
    valids = opt.val_ids
    featurelist0 = os.listdir(os.path.join(opt.dataroot, opt.modality_filename[0]))
    featurelist0 = natsort.natsorted(featurelist0)
    featurelist = featurelist0[testids[0]:testids[1]]
    for cube in featurelist:
        test_images[0, :, :, :] = np.load(os.path.join(opt.feature_dir, cube + '.npy'))
        images = torch.from_numpy(test_images)
        images = images.to(device=device, dtype=torch.float32)
        pred,featuremap= net(images)
        pred = torch.nn.functional.softmax(pred, dim=1)
        result=pred[0,1, :,:].cpu().detach().numpy()*255
        imageio.imsave(os.path.join(test_results, cube + ".bmp"), result.astype(np.uint8))  
        print(cube)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    opt = TestOptions().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = model_LANet.UNet_3Plus(in_channels=1, channels=opt.plane_perceptron_channels, n_classes=opt.n_classes)
    restore_path = 'logs/best_model_V2+2/0.935200/1200.pth'
    net.load_state_dict(
        torch.load(restore_path, map_location=device)
    )
    net.to(device=device)
    try:
        test_net(net=net,device=device)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
