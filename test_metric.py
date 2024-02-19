from utils import *
import os
import natsort
import imageio
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import cv2
import cldice
import matplotlib.pylab as plt
import seaborn as sns


def comp_metrics(pred_dir,label_dir):
    ctList=os.listdir(pred_dir)
    dic, pre, rec, jac, bacc, cldic = [],[],[],[],[],[]
    soft_para = 0.5
    cl = cldice.soft_cldice()

    for ct in tqdm(ctList):
        ctPath = os.path.join(pred_dir, ct)
        labelPath = os.path.join(label_dir, ct)  
        image= imageio.imread(ctPath)
        image = np.where((image/255)>soft_para, 1, 0).astype(np.float32)
        label= imageio.imread(labelPath).astype(np.float32)
        dice = cal_Dice(image,label)
        dic.append(dice)
        prec, reca, jacc, baccu = cal_PRE_REC_JAC_BACC(image,label)
        pre.append(prec)
        rec.append(reca)
        jac.append(jacc)
        bacc.append(baccu)
        cld = cl(torch.tensor(label).unsqueeze(0).unsqueeze(0),torch.tensor(image).unsqueeze(0).unsqueeze(0))
        cldic.append(cld)

    return dic, pre, rec, jac, bacc, cldic

def computeQualityMeasures(lP,lT):
    quality=dict()
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()
 
    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["dice"]=dicecomputer.GetDiceCoefficient()
 
    return quality


def dic_VS(dir1='', dir2='', label_dir = ''):
    dic1,_,_,_,_,cldice1 = comp_metrics(dir1,label_dir)
    dic2,_,_,_,_,cldice2 = comp_metrics(dir2,label_dir)
    dic1 = np.array(dic1)
    dic2 = np.array(dic2)
    compare = dic2-dic1
    tmp = list(map(list,zip(range(len(compare)), compare))) 
    large = sorted(tmp,key=lambda x:x[1],reverse=True) 
    print(large[:5])



def violin_plot():
    data_path = 'logs/dice_6M'
    data = []
    npyList=os.listdir(data_path)
    for npy in npyList:
        loadData = np.load(os.path.join(data_path,npy))
        data.append(loadData)

    plt.figure(dpi = 300)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    label = ["IPN", "IPN-V2", "IPN-V2+", 'LA-Net', 'LA-Net+']
    font_1 = {"size": 12}

    sns.violinplot(data)
    plt.xlabel("Methods", font_1)
    plt.ylabel("DICE (%)", font_1)
    plt.xticks(ticks = [0, 1, 2, 3, 4], labels = label, fontsize = 8)
    plt.yticks(fontsize = 8)

    plt.savefig('violin-6M.png')


def feature_projection2D():
    data_path = '/data2/SkipFeature/IPNV2-6M/x4'
    save_path = 'logs/ComposeMap/skip/IPNV2-6M/x4'
    for npy in tqdm(os.listdir(data_path)):
        path = os.path.join(data_path, npy)
        projection = np.load(path)

        ave = np.average(projection,0)
        max = np.max(projection,0)
               
        heatmap_ave = cv2.normalize(ave, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_max = cv2.normalize(max, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        imageio.imsave(os.path.join(save_path, 'ave', npy[:-4] + ".bmp"), heatmap_ave.astype(np.uint8))
        imageio.imsave(os.path.join(save_path, 'max', npy[:-4] + ".bmp"), heatmap_max.astype(np.uint8))


def overlap_pic():
    fig1_dir = '/data2/datasets/OCTA-500/6M/Label_RV'
    fig2_dir = 'logs/ComposeMap/skip/LA-Net-6M/x4'
    save_dir = 'logs/ComposeMap/skip/LA-Net-6M/heatmap/x4'
    for n in tqdm(os.listdir(fig2_dir)):
        for name in os.listdir(os.path.join(fig2_dir, n)):
            fig1_path = os.path.join(fig1_dir, name)
            fig2_path = os.path.join(fig2_dir, n, name)

            fig1 = cv2.imread(fig1_path,0)  
            fig1 = np.where(fig1>0,255,fig1)
            fig2 = cv2.imread(fig2_path,0)
            fig2 = 255 - fig2
            heatmap = cv2.applyColorMap(fig2,2)
            combine = cv2.addWeighted(cv2.cvtColor(fig1, cv2.COLOR_GRAY2RGB),0.5,heatmap,0.5,0)
            imageio.imsave(os.path.join(save_dir, n, "comb-"+name), combine)



if __name__ == '__main__':
    pred_dir = 'logs/test_results'
    label_dir = '/data2/datasets/OCTA-500/6M/Label_RV'

    dic, pre, rec, jac, bacc, cldic = comp_metrics(pred_dir,label_dir)
    print('\ndic =',np.mean(dic),'\t std =',np.std(dic))
    print('\npre =',np.mean(pre),'\t std =',np.std(pre))
    print('\nrec =',np.mean(rec),'\t std =',np.std(rec))
    print('\njac =',np.mean(jac),'\t std =',np.std(jac))
    print('\nbacc =',np.mean(bacc),'\t std =',np.std(bacc))
    
