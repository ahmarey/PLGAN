#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:07:43 2021

@author: dronet
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:10:46 2020

@author: amarey
"""

import tensorflow as tf
import numpy as np
import os
import time
import cv2
from matplotlib import pyplot as plt
from IPython import display

os.environ["CUDA_VISIBLE_DEVICES"]="3" #comment or un comment to decide to use which GPU

main_dir="./" 



height = 40



CHANNEL=2
scale=1.07
negative = 24
PATH = '900Mhz_40m/'
shift=0

#nloss_ndisc_adam_nGen_adam_trainUP_Work_NEW
#900Mhz_40m_satellite_model_nloss_ndisc_adam_nGen_adam_trainUP_Work_NEW_indoor
SETTINGS_sat = 'nloss_ndisc_adam_nGen_adam_trainUP_Work_NEW_indoor'
sat_check=main_dir  + PATH[:-1] + '_satellite_model_' + SETTINGS_sat
# sat_check='C:/Users/amarey/Desktop/ahmed/Power_Trio/900Mhz_300m/heightmap nLoss nDisc_adam oGen_model/'
# sat_check=main_dir+PATH+PATH[:-1]+'_satellite_model_oloss_ndisc_adam_nGen_adam_trainUP_Work'
# hmap_check='C:/Users/amarey/Desktop/ahmed/Power_Trio/900Mhz_300m/heightmap_model/'

hmap_check=main_dir+PATH[:-1]+'_heightmap_model_nloss_ndisc_adam_nGen_adam_trainUP_Work_NEW'

SETTINGS = 'ndisc_ngen_adam'
INPUTFOLDER='trainUP/'
INPUTFOLDER2='test/'
test_size=len(os.listdir(main_dir+INPUTFOLDER2))

generatorSAT = tf.saved_model.load(sat_check)
generatorHEIGHT = tf.saved_model.load(hmap_check) 
    

BUFFER_SIZE = 400
BATCH_SIZE =  32#@param {type:"integer"}

def nping(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg
def load(image_file):
  # print('load')
  # print('image_file:',image_file)
  # image = tf.io.read_file(image_file)
  # image = tf.image.decode_bmp(image)
  # path = tf.keras.utils.get_file(os.path.basename(image_file), image_file)
  # you should decode bytes type to string type
  if(type(image_file)!=str):
    image_file=image_file.numpy()

  # print("file_path: ",image_file)
  image=np.load(image_file)
  # image=nping(image)
  # print('after loading')
  w = tf.shape(image)[1]
  w = w // 3

  sat_image = image[:, :w, :]
  # real_image = image[:, 2*w:, :]
  input_image = image[:, w:2*w, :]  #ıtis actually heıghtmap
  real_image = (image[:, 2*w:, :]+negative)/scale
#  if (np.count_nonzero(real_image<0)):
#    print(image_file)
##    print('====================================================================================================================================')
##    print('====================================================================================================================================')
#    print(real_image.min())
#    # break
#  if(np.count_nonzero(real_image>255)):
#    print(image_file)
##    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
##    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#    print(real_image.max())
##    print(real_image.min())

    
  
  

  name=nping(int(os.path.basename(image_file)[:-4]))  
  sat_image = tf.cast(sat_image, tf.float32)
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image, sat_image, name



def normalize(input_image, real_image,sat_image):
  sat_image=(sat_image/ 127.5) - 1
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image,sat_image







def load_image_test(image_file):

  input_image, real_image,sat_image,file_name = load(image_file)
  input_image, real_image,sat_image = normalize(input_image, real_image, sat_image)

  return input_image, real_image, sat_image, file_name


test_dataset = tf.data.Dataset.list_files(main_dir+INPUTFOLDER2+'*.npy')
test_dataset = test_dataset.map(lambda x: tf.py_function(load_image_test, [x], [tf.float32,tf.float32,tf.float32,tf.float32]))

test_dataset = test_dataset.batch(BATCH_SIZE)


OUTPUT_CHANNELS = 3



# Real PART
from sklearn.metrics import mean_squared_error
import numpy.ma as ma

MSEsHMAP=[]
HIST_MSE8sHMAP=[]
HIST_MSE256sHMAP=[]
MSEs_outHMAP=[]
HIST_MSE8s_outHMAP=[]
HIST_MSE256s_outHMAP=[]
MSEsSAT=[]
HIST_MSE8sSAT=[]
HIST_MSE256sSAT=[]
MSEs_outSAT=[]
HIST_MSE8s_outSAT=[]
HIST_MSE256s_outSAT=[]
test_results=np.zeros((test_size,7))
counting=[]
var_calc=np.zeros((256,256,test_size))
var_hist=np.zeros((299,test_size))


# plt.rcParams.update({'font.size': 30})
trying_bins8=np.zeros(9)
for i in range(9):
  trying_bins8[i]=-30+i*35
  
y_ticks=np.zeros(7)
for i in range(7):
    y_ticks[i]=i*0.05
  
trying_ticks8=np.zeros(7)
for i in range(7):
  trying_ticks8[i]=-25+i*50



trying_bins256=np.zeros(300)
for i in range(300):
  trying_bins256[i]=i-25


def mse(imageA, imageB):

	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

def outdoor_mse(heightmap,original_pl,predicted_pl):
  denom=heightmap.sum()
  pred_outdoor_pl=heightmap*predicted_pl
  original_pl=original_pl*heightmap
  error=original_pl-pred_outdoor_pl
  sq_error=error**2
  mse_outdoor=np.sum(sq_error)/denom
  return mse_outdoor


def masking(pic,num):
  masked_pic=ma.masked_equal(pic,num)  
  cmpres_pic=masked_pic.compressed()
  return cmpres_pic

def histing(vector,bin,maximum,offset):

  hist,binedge=np.histogram(vector,bins=bin,density=False)
  hist=hist/np.sum(hist)


  return hist
max_prob=np.zeros(6)
  #%%  
def generate_images(modelhmap, modelsat, test_input, tar, sats, name,count):
    
  
  print(test_input.shape,count)
  predictionsat = modelsat(sats, training=False)
  predictionhmap = modelhmap(test_input, training=False)
  counter=32*count
  print(test_input.shape[0])
  maximum=255
  offset=0   
 
  for i in range(test_input.shape[0]):
#  for i in range(2):

    sat=np.array(sats[i])
    predict_plsat = np.array(predictionsat[i])

    sat=sat* 0.5 + 0.5
    sat= sat *255
  
    counter = 32*count+i
    counting.append(counter)
    
    
    hmap1 = np.array(test_input[i])
    true_pl = np.array(tar[i])
    
    


    name=np.array(file_name[i])

    hmap1=hmap1[...,CHANNEL] * 0.5 + 0.5
    hmap1=hmap1*255
    true_pl=true_pl[...,CHANNEL] * 0.5 + 0.5
    true_pl=true_pl*255
    var_calc[...,counter] = true_pl
    hmap=(hmap1==0).astype(int)
    
    trial_outdoor_true_pl=(hmap*true_pl*scale-negative-shift)
    outdoor_true_pl=hmap*(true_pl*scale-negative-shift)
    cmpres_outdoor_true=masking(outdoor_true_pl.flatten(),0)
    hist_true256out=histing(cmpres_outdoor_true,trying_bins256,maximum,offset)
    hist_true8out=histing(cmpres_outdoor_true,trying_bins8,maximum,offset)
    
    hist_true256 = histing(true_pl, trying_bins256,maximum,offset)
    hist_true8 = histing(true_pl, trying_bins8, maximum, offset)
    
    
    
    predict_plsat = np.array(predictionsat[i])
    predict_plsat = predict_plsat[...,CHANNEL] * 0.5 + 0.5
    predict_plsat = predict_plsat*255
    
    trial_show_outdoor_predict_plsat = (hmap*predict_plsat*scale-negative-shift)
    show_outdoor_predict_plsat = hmap*(predict_plsat*scale-negative-shift)
    cmpres_outdoor_predsat = masking(show_outdoor_predict_plsat.flatten(),0)
    hist_pred256satout = histing(cmpres_outdoor_predsat, trying_bins256, maximum,offset)
    hist_pred8satout = histing(cmpres_outdoor_predsat, trying_bins8, maximum,offset)
    hist_pred256sat = histing(predict_plsat, trying_bins256,maximum,offset)
    hist_pred8sat = histing(predict_plsat, trying_bins8,maximum,offset)  
    mse_outdoorsat=outdoor_mse(hmap,true_pl,predict_plsat)
    
    hist_mse256satout = mean_squared_error(hist_true256out, hist_pred256satout)      
    hist_mse8satout = mean_squared_error(hist_true8out, hist_pred8satout)    
    hist_mse256sat = mean_squared_error(hist_true256, hist_pred256sat)      
    hist_mse8sat = mean_squared_error(hist_true8, hist_pred8sat)
    var_hist[...,i]=hist_true256out
    
    mse_allsat=mean_squared_error(true_pl,predict_plsat)    
    MSEsSAT.append(mse_allsat)
    HIST_MSE8sSAT.append(hist_mse8sat)
    HIST_MSE256sSAT.append(hist_mse256sat)
    MSEs_outSAT.append(mse_outdoorsat)
    HIST_MSE8s_outSAT.append(hist_mse8satout)
    HIST_MSE256s_outSAT.append(hist_mse256satout)
    
    
    
    predict_plhmap = np.array(predictionhmap[i])
    predict_plhmap=predict_plhmap[...,CHANNEL] * 0.5 + 0.5
    predict_plhmap=predict_plhmap*255
    
    trial_show_outdoor_predict_plhmap =( hmap*predict_plhmap*scale-negative-shift)
    show_outdoor_predict_plhmap=hmap*(predict_plhmap*scale-negative-shift)
    cmpres_outdoor_predhmap=masking(show_outdoor_predict_plhmap.flatten(),0)
    
    hist_pred256hmapout = histing(cmpres_outdoor_predhmap,trying_bins256,maximum,offset)
    hist_pred8hmapout = histing(cmpres_outdoor_predhmap,trying_bins8,maximum,offset)
    hist_pred256hmap = histing(predict_plhmap, trying_bins256,maximum,offset)
    hist_pred8hmap = histing(predict_plhmap, trying_bins8,maximum,offset)   
    mse_outdoorhmap=outdoor_mse(hmap,true_pl,predict_plhmap)
    
    hist_mse256hmapout = mean_squared_error(hist_true256out,hist_pred256hmapout)  
    hist_mse8hmapout = mean_squared_error(hist_true8out,hist_pred8hmapout)   
    hist_mse256hmap = mean_squared_error(hist_true256, hist_pred256hmap)      
    hist_mse8hmap = mean_squared_error(hist_true8,hist_pred8hmap)      
    
    mse_allhmap=mean_squared_error(true_pl,predict_plhmap)    
    MSEsHMAP.append(mse_allhmap)
    HIST_MSE8sHMAP.append(hist_mse8hmap)
    HIST_MSE256sHMAP.append(hist_mse256hmap)
    MSEs_outHMAP.append(mse_outdoorhmap)
    HIST_MSE8s_outHMAP.append(hist_mse8hmapout)
    HIST_MSE256s_outHMAP.append(hist_mse256hmapout)    
    denorm_height=(hmap1)
    denormheightmax=denorm_height.max()

    trial_sum=np.sum(show_outdoor_predict_plhmap)
    trial_sum2=np.sum(cmpres_outdoor_predhmap)
    
    trial_sum=np.sum(show_outdoor_predict_plsat)
    trial_sum2=np.sum(cmpres_outdoor_predsat)

    
   
    
    if(max_prob[1] < hist_pred256hmapout.max()):
        max_prob[1] = hist_pred256hmapout.max()
        max_prob[0] = int(name)
    if(max_prob[3] < hist_pred256satout.max()):
       max_prob[3] = hist_pred256satout.max()
       max_prob[2] = int(name)
    if(max_prob[5] < hist_true256out.max()):
       max_prob[5] = hist_true256out.max()
       max_prob[4] = int(name)
    
    
   
    
    
    test_results[counter,0] = int(name)
    test_results[counter,1] = mse_outdoorhmap
    test_results[counter,2] = hist_mse8hmapout
    test_results[counter,3] = hist_mse256hmapout
    test_results[counter,4] = mse_outdoorsat
    test_results[counter,5] = hist_mse8satout
    test_results[counter,6] = hist_mse256satout
#    
    

    
    '''
    heightmap input
    '''
    fig_size=20
    fig1, ax1 = plt.subplots(figsize=(fig_size,fig_size))
    im1=ax1.imshow(denorm_height/denormheightmax*381,vmin=0,vmax=400,cmap='RdBu_r')
    ax1.axis('off')
    ax1.set_xlabel('Heightmap input ')

    xx1=fig1.colorbar(im1, ax=ax1,shrink=0.67)    
    xx1.ax.tick_params(labelsize=50)
    fig1.savefig( main_dir+'predictionsnow_sep/'+str(int(name))+'hmap.pdf',bbox_inches='tight')
    fig1.tight_layout()
    plt.close()
    
    '''
      Satellite input
    '''
    fig4, ax4 = plt.subplots(figsize=(fig_size,fig_size))
    ax4.imshow(sat/255)
    ax4.axis('off')
#    ax4.set_xlabel('Satellite input ')
    fig4.savefig( main_dir+'predictionsnow_sep/'+str(int(name))+'sat.pdf',bbox_inches='tight')
    plt.tight_layout()
    plt.close()
    


    
    '''
      true path loss in im3 and im3
    '''
    fig3, ax3 = plt.subplots(figsize=(fig_size,fig_size))
    im3=ax3.imshow((trial_outdoor_true_pl-shift), vmin=-25, vmax=275, cmap='RdBu_r')
    ax3.axis('off')

    xx3=fig1.colorbar(im3, ax=ax3,shrink=0.67, ticks=[-25, 25, 75,125,175,225,275])
    xx3.ax.tick_params(labelsize=55)
    fig3.savefig( main_dir+'predictionsnow_sep/'+str(int(name))+'true.pdf',bbox_inches='tight')
    plt.tight_layout()
    plt.close()
    

    fig2, ax2 = plt.subplots(figsize=(fig_size,fig_size))
    im2=ax2.imshow((trial_show_outdoor_predict_plhmap-shift),vmin=-25,vmax=275,cmap='RdBu_r')
    ax2.axis('off')
    ax2.set_xlabel('Heightmap prediction',fontsize = 70)
    xx2=fig1.colorbar(im2, ax=ax2,shrink=0.67, ticks=[-25, 25, 75,125,175,225,275])
    xx2.ax.tick_params(labelsize=55)
    fig2.savefig( main_dir+'predictionsnow_sep/'+str(int(name))+'hmap_pred.pdf',bbox_inches='tight')
    plt.tight_layout()
    plt.close()
    

    fig5, ax5 = plt.subplots(figsize=(fig_size,fig_size))
    im5=ax5.imshow((trial_show_outdoor_predict_plsat-shift),vmin=-25,vmax=275,cmap='RdBu_r')
    ax5.axis('off')
    xx5=fig1.colorbar(im5, ax=ax5, shrink=0.67, ticks=[-25, 25, 75,125,175,225,275])
    xx5.ax.tick_params(labelsize=55)
    fig5.savefig( main_dir+'predictionsnow_sep/'+str(int(name))+'sat_pred.pdf',bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    '''
      HISTOGRAM COMPARISON FOR 300 BINS
    '''

    
    plt.figure(figsize=[60, 40]) 
    plt.plot(trying_bins256[:-1], hist_true256out,label='True Distribution',linewidth=12)
    plt.plot(trying_bins256[:-1],  (hist_pred256satout),label='Satellite Predicted Distribution',linewidth=12)
    plt.plot(trying_bins256[:-1],  (hist_pred256hmapout),label='Height Map Predicted Distribution',linewidth=12)

    plt.xticks(ticks=trying_ticks8,fontsize=175)
    plt.yticks(ticks=y_ticks,fontsize=175)
    plt.xlabel('Excessive Path Loss (dB)',fontsize=175)
    plt.ylabel('Probability',fontsize=175)
    plt.ylim((0,0.1))

    plt.grid(axis='y', alpha=10)
   
    plt.tick_params(direction='out', length=10, width=4, grid_alpha=0.5)    
    plt.legend(loc='upper right', shadow=False, fontsize=175)
    plt.tight_layout()
    plt.savefig( main_dir+'predictionsnow_sep/'+str(int(name))+'_300.pdf') 
    plt.close()
  print(max_prob)
count=0
    
    
    


for example_input, example_target, example_sat, file_name in test_dataset:
  generate_images(generatorHEIGHT,generatorSAT, example_input.numpy(), example_target.numpy(), example_sat.numpy(),file_name.numpy(),count)
  count=count+1
#%%  

sorted_test_results=test_results[test_results[:,0].argsort()]
File_object1 = open(main_dir+'predictionsnow_sep/results.txt',"a")
File_object1.write( ' name  msehmap    8bins    280bins  mseSAT    8bins    280bins \n')
for i in range(test_results.shape[0]):
    File_object1.write( ' {:4d} {:4d}     {:.2e}  {:.2e}  {:4d}    {:.2e}    {:.2e}  \n'
                      .format(int(sorted_test_results[i,0]), int(sorted_test_results[i,1])
                              ,sorted_test_results[i,2],sorted_test_results[i,3]
                              ,int(sorted_test_results[i,4])
                              ,sorted_test_results[i,5],sorted_test_results[i,6]))
    
             
File_object1.close()
variance = np.var(var_calc,dtype=np.float64)
variance_hist = np.var(var_hist,dtype=np.float64)
File_object = open(main_dir+'predictionsnow_sep/results_all.txt',"a")
File_object.write('these are the result of dataset {} with the following settings {} \n'
                  .format(PATH,SETTINGS))

# File_object.write('Min MSE all: {:.2e} Max: {:.2e}\n'.format(min(MSEsHMAP),max(MSEsHMAP)))
File_object.write('MSE_ALL   HMAP Min: {:.2e} Max: {:.2e} AVG: {:.2e}\n'
                  .format(min(MSEsHMAP), max(MSEsHMAP),np.average(np.array(MSEsHMAP))))

File_object.write('MSE_ALL   SAT  Min: {:.2e} Max: {:.2e} AVG: {:.2e}\n\n'
                  .format(min(MSEsSAT), max(MSEsSAT),np.average(np.array(MSEsSAT))))








# print('Min Histogram MSE8 heightmap nLoss nDisc_adam oGen_adam all: {:.2e}'.format(min(HIST_MSE8sSAT)),'Max: {:.2e}'.format(max(HIST_MSE8sSAT)))
File_object.write('Variance of pathloss images is :            {:.2e}\n'
                  .format(variance))
File_object.write('Variance of pathloss histogram is :            {}\n'
                  .format(variance_hist))
File_object.write('MSE8ALL   HMAP Min: {:.2e} Max: {:.2e} AVG: {:.2e}\n'
                  .format(min(HIST_MSE8sHMAP),max(HIST_MSE8sHMAP),np.average(np.array(HIST_MSE8sHMAP))))
File_object.write('MSE8ALL   SAT  Min: {:.2e} Max: {:.2e} AVG: {:.2e}\n\n'
                  .format(min(HIST_MSE8sSAT),max(HIST_MSE8sSAT),np.average(np.array(HIST_MSE8sSAT))))


File_object.write('MSE256ALL HMAP Min: {:.2e} Max: {:.2e} AVG: {:.2e}\n'
                  .format(min(HIST_MSE256sHMAP),max(HIST_MSE256sHMAP),np.average(np.array(HIST_MSE256sHMAP))))
File_object.write('MSE256ALL SAT  Min: {:.2e} Max: {:.2e} AVG: {:.2e}\n\n\n'
                  .format(min(HIST_MSE256sSAT),max(HIST_MSE256sSAT),np.average(np.array(HIST_MSE256sSAT))))



File_object.write('MSE_OUT   HMAP Min: {:.2e} Max: {:.2e} AVG: {:.2e}\n'
                  .format(min(MSEs_outHMAP), max(MSEs_outHMAP),np.average(np.array(MSEs_outHMAP))))

File_object.write('MSE_OUT   SAT  Min: {:.2e} Max: {:.2e} AVG: {:.2e}\n\n'
                  .format(min(MSEs_outSAT), max(MSEs_outSAT),np.average(np.array(MSEs_outSAT))))


File_object.write('MSE8OUT   HMAP Min: {:.2e} Max: {:.2e} AVG: {:.2e}\n'
                  .format(min(HIST_MSE8s_outHMAP),max(HIST_MSE8s_outHMAP),np.average(np.array(HIST_MSE8s_outHMAP))))
File_object.write('MSE8OUT   SAT  Min: {:.2e} Max: {:.2e} AVG: {:.2e}\n\n'
                  .format(min(HIST_MSE8s_outSAT),max(HIST_MSE8s_outSAT),np.average(np.array(HIST_MSE8s_outSAT))))


File_object.write('MSE256OUT  HMAP Min: {:.2e} Max: {:.2e} AVG: {:.2e}\n'
                  .format(min(HIST_MSE256s_outHMAP),max(HIST_MSE256s_outHMAP),np.average(np.array(HIST_MSE256s_outHMAP))))
File_object.write('MSE256OUT  SAT  Min: {:.2e} Max: {:.2e} AVG: {:.2e}\n\n\n'
                  .format(min(HIST_MSE256s_outSAT),max(HIST_MSE256s_outSAT),np.average(np.array(HIST_MSE256s_outSAT))))



File_object.write('max probability HMAP :{} in sample {}\n'.format(max_prob[1].item(),int(max_prob[0])))
File_object.write('max probability TRUE :{} in sample {}\n'.format(max_prob[5].item(),int(max_prob[4])))
File_object.write('max probability SAT  :{} in sample {}\n\n\n'.format(max_prob[3].item(),int(max_prob[2])))


File_object.close()
