import imageio
import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
import model_LANet
# import model_IPN_V2
import utils
import shutil
import natsort
from options.train_options import TrainOptions
from skimage import transform
import data_process.readData as readData
import data_process.BatchDataReader as BatchDataReader
import time


def train_net(net,device):
    interval=opt.save_interval
    train_num = opt.train_ids[1] - opt.train_ids[0]
    val_num = opt.val_ids[1] - opt.val_ids[0]
    DATA_SIZE = opt.data_size
    BLOCK_SIZE = opt.block_size
    val_images = np.zeros((1, opt.in_channels, BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[2]))
    cube_images = np.zeros((1, opt.in_channels, BLOCK_SIZE[0], DATA_SIZE[1], DATA_SIZE[2]))
    best_valid_dice=0
    model_save_path = os.path.join(opt.saveroot, 'checkpoints')
    best_model_save_path = os.path.join(opt.saveroot, 'best_model')

    print("Start Setup dataset reader")
    train_records, validation_records = readData.read_dataset(opt.dataroot, opt.train_ids, opt.val_ids, opt.modality_filename)  
    print("Setting up dataset reader")
    train_dataset_reader = BatchDataReader.BatchDatset(train_records, opt.modality_filename,opt.data_size, opt.block_size, opt.in_channels,
                                                       opt.batch_size, train_num, "train", opt.saveroot)  

    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)
    criterion = nn.CrossEntropyLoss()

    for itr in range(0, opt.max_iteration):
        net.train()
        train_images, train_annotations = train_dataset_reader.read_batch_random_train()  

        train_images =train_images.to(device=device, dtype=torch.float32)
        train_annotations = train_annotations.to(device=device, dtype=torch.long)
        pred,_= net(train_images)
        loss = criterion(pred, train_annotations)  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if itr % 10 == 0:
            print(itr,loss.item())
        with torch.no_grad():
            if (itr % interval==0):
                torch.save(net.module.state_dict(),
                           os.path.join(model_save_path,f'{itr}.pth'))
                logging.info(f'Checkpoint {itr} saved !')
                val_Dice_sum = 0
                net.eval()
                valids = opt.val_ids
                cubelist0 = os.listdir(os.path.join(opt.dataroot, opt.modality_filename[0]))
                cubelist0 = natsort.natsorted(cubelist0)
                cubelist = cubelist0[valids[0]:valids[1]]
                for kk, cube in enumerate(cubelist):
                    bscanlist = os.listdir(os.path.join(opt.dataroot, opt.modality_filename[0], cube))
                    bscanlist = natsort.natsorted(bscanlist)
                    for i, bscan in enumerate(bscanlist):
                        for j, modal in enumerate(opt.modality_filename):
                            if modal != opt.modality_filename[-1]:
                                cube_images[0, j, :, :, i] = np.array(transform.resize(imageio.imread(os.path.join(opt.dataroot, modal, cube, bscan)),[BLOCK_SIZE[0], DATA_SIZE[1]], order=1, preserve_range=True))  
                    result = np.zeros((DATA_SIZE[1], DATA_SIZE[2]))
                    label= imageio.imread(os.path.join(opt.dataroot, opt.modality_filename[opt.in_channels], f'{cube}.bmp'))
                    for i in range(0, DATA_SIZE[1],BLOCK_SIZE[1]):
                        for j in range(0, DATA_SIZE[2],BLOCK_SIZE[2] ):
                            val_images[0, :, 0:BLOCK_SIZE[0], 0:BLOCK_SIZE[1], 0:BLOCK_SIZE[2]] = cube_images[0, :, :,i:i + BLOCK_SIZE[1],j:j + BLOCK_SIZE[2]]
                            images = torch.from_numpy(val_images)
                            images = images.to(device=device, dtype=torch.float32)
                            pred,_ = net(images)
                            pred_argmax = torch.argmax(pred, dim=1)
                            result[i:i + BLOCK_SIZE[1], j:j + BLOCK_SIZE[2]] = result[i:i + BLOCK_SIZE[1],j:j + BLOCK_SIZE[2]] + pred_argmax[0, 0, :, :].cpu().detach().numpy()
                    val_Dice_sum+= utils.cal_Dice(result,label)
                val_Dice=val_Dice_sum/val_num
                print("Step:{}, Valid_Dice:{}".format(itr,val_Dice))

                if val_Dice > best_valid_dice:
                    temp = '{:.5f}'.format(val_Dice)
                    if not os.path.exists(os.path.join(best_model_save_path,temp)):
                        os.mkdir(os.path.join(best_model_save_path,temp))
                    temp2= f'{itr}.pth'
                    shutil.copy(os.path.join(model_save_path,temp2),os.path.join(best_model_save_path,temp,temp2))
                    model_names = natsort.natsorted(os.listdir(best_model_save_path))
                    if len(model_names) == 4:
                        shutil.rmtree(os.path.join(best_model_save_path,model_names[0]))
                    best_valid_dice = val_Dice


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    opt = TrainOptions().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = model_LANet.LA_Net(in_channels=opt.in_channels, out_channels=opt.channels,plane_perceptron_channels=opt.plane_perceptron_channels, n_classes=opt.n_classes,block_size=opt.block_size)
    
    net=torch.nn.DataParallel(net).cuda()  

    if opt.load:
        net.load_state_dict(
            torch.load(opt.load, map_location=device)
        )
        logging.info(f'Model loaded from {opt.load}')
    try:
        t0 = time.time()
        train_net(net=net,device=device)
        print("It takes ", (time.time() - t0) / 60 / 60, "hours.")
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
