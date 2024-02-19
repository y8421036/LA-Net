import cv2
import torch
import torch.nn as nn
import logging
import sys
import os
import model_LANet
import model_IPN_V2
import numpy as np
from options.test_options import TestOptions
import natsort
from scipy import io
from PIL import Image
import imageio

def test_net(net,device):
    DATA_SIZE = opt.data_size
    BLOCK_SIZE = opt.block_size
    net.eval()
    test_images = np.zeros((1, opt.in_channels, BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[2]))
    cube_images = np.zeros((1, opt.in_channels, BLOCK_SIZE[0], DATA_SIZE[1], DATA_SIZE[2]))

    modalitylist = opt.modality_filename
    testids = opt.test_ids
    valids = opt.val_ids
    trainids= opt.train_ids
    cubelist0 = os.listdir(os.path.join(opt.dataroot, modalitylist[0]))
    cubelist0 = natsort.natsorted(cubelist0)
    if npy == 1:
        cubelist =cubelist0[trainids[0]:trainids[1]]+cubelist0[valids[0]:valids[1]]+cubelist0[testids[0]:testids[1]]
    elif npy == 2 or npy == 3 or npy == 4:
        cubelist = cubelist0[testids[0]:testids[1]]

    with torch.no_grad():
        vote_time=4 
        for kk,cube in enumerate(cubelist):
            bscanlist = os.listdir(os.path.join(opt.dataroot, modalitylist[0], cube))
            bscanlist=natsort.natsorted(bscanlist)
            for i,bscan in enumerate(bscanlist):
                for j,modal in enumerate(modalitylist):
                    if modal!=opt.modality_filename[-1]:
                        temp_image = Image.open(os.path.join(opt.dataroot,modal,cube,bscan)) 
                        temp_image = temp_image.resize(([DATA_SIZE[1], BLOCK_SIZE[0]]), Image.BILINEAR) 
                        temp_image = np.array(temp_image)
                        cube_images[0,j,:,:,i]=temp_image

            result =np.zeros((DATA_SIZE[1], DATA_SIZE[2]))
            projection_2D_64c = np.zeros((opt.channels, DATA_SIZE[1], DATA_SIZE[2])) 
            skip1 = np.zeros((opt.channels, DATA_SIZE[1], DATA_SIZE[2])) 
            votemap=np.zeros((DATA_SIZE[1], DATA_SIZE[2]))  
            votemap_projection = np.zeros((opt.channels, DATA_SIZE[1], DATA_SIZE[2]))  

            for i in range(0,DATA_SIZE[1]-BLOCK_SIZE[1]+BLOCK_SIZE[1]//vote_time,BLOCK_SIZE[1]//vote_time):
                for j in range(0,DATA_SIZE[2]-BLOCK_SIZE[2]+BLOCK_SIZE[2]//vote_time,BLOCK_SIZE[2]//vote_time):
                    test_images[0, :, 0:BLOCK_SIZE[0], 0:BLOCK_SIZE[1], 0:BLOCK_SIZE[2]] = cube_images[0, :, :,i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]
                    images = torch.from_numpy(test_images)
                    images = images.to(device=device, dtype=torch.float32)
                    if npy == 1 or npy ==2:
                        pred,_ = net(images)                 
                        pred = torch.nn.functional.softmax(pred, dim=1)  
                        votemap[i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]=votemap[i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]+1
                        result[i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]] = result[i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]+pred[0,1,0,:,:].cpu().detach().numpy()  
                    elif npy == 3:
                        pred,_,projection = net(images)
                        votemap_projection[:, i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]] = votemap_projection[:, i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]+1
                        projection_2D_64c[:, i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]] = projection_2D_64c[:, i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]+projection[0,:,0,:,:].cpu().detach().numpy()
                    elif npy == 4:
                        pred,_,x1 = net(images)
                        votemap_projection[:, i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]] = votemap_projection[:, i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]+1
                        skip1[:, i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]] = skip1[:, i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]+x1[0,:,:,:].cpu().detach().numpy()
                        
            
            print(cube)
            if npy == 1:
                result=result/votemap*255 
                np.save(os.path.join(feature_results_path, cube + ".npy"),result)
            elif npy ==2:
                result=result/votemap*255 
                imageio.imsave(os.path.join(test_results_path, cube + ".bmp"), result.astype(np.uint8))
            elif npy ==3:
                projection_2D_64c = projection_2D_64c/votemap_projection
                np.save(os.path.join(projection_save_path, cube + ".npy"),projection_2D_64c)
            elif npy ==4:
                skip1=skip1/votemap_projection
                
                np.save(os.path.join(skip_save_path, 'x1', cube + ".npy"),skip1)
                



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    opt = TestOptions().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    npy = 1  
    restore_path = 'logs/best_model2/0.92883/12300.pth'
    feature_results_path= opt.feature_dir  
    test_results_path = os.path.join(opt.saveroot, 'test_results')  
    projection_save_path = 'logs/ProjectionFeature2D/IPNV2-6M'  
    skip_save_path = '/data2/SkipFeature/IPNV2-6M'

    net = model_LANet.LA_Net(in_channels=opt.in_channels, out_channels=opt.channels,plane_perceptron_channels=opt.plane_perceptron_channels, n_classes=opt.n_classes,
                           block_size=opt.block_size)
    
    
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
