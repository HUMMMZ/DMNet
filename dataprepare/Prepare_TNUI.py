# -*- coding: utf-8 -*-
##scipy==1.2.1

import h5py
import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob
import imageio
# Parameters
height = 256 # Enter the image size of the model.
width  = 256 # Enter the image size of the model.
channels = 3 # Number of image channels

train_number = 966  # Randomly assign the number of images for generating the training set.
val_number = 276   # Randomly assign the number of images for generating the validation set.
test_number = 139  # Randomly assign the number of images for generating the test set.
Dataset_add = './TNUI-2021-/thyroid_data/'
Tr_add = 'train/images'

Tr_list = glob.glob(Dataset_add+ Tr_add+'/*.png')

# It contains 2000 training samples
Data_train_2017    = np.zeros([train_number, height, width, channels])
Label_train_2017   = np.zeros([train_number, height, width])

print('Reading TN3K Training')
print(Tr_list)
for idx in range(len(Tr_list)):
    print(idx+1)
    img = sc.imread(Tr_list[idx], mode = 'RGB')
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    Data_train_2017[idx, :,:,:] = img

    b = Tr_list[idx] 
    #print('b:',b)
    a = b[0:32]
    #print('a:',a)
    b = b.split('.')[1].split('/')[5]
    #print('b2:',b)
    add = (a+ 'masks/' + b +'.png')    
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train_2017[idx, :,:] = img2    
         
print('Reading TN3K Training finished')

#############################################################################
#############################################################################
###########################Validation_img_read###############################
#############################################################################
#############################################################################
Val_add = 'val/images'

Val_list = glob.glob(Dataset_add+ Val_add+'/*.png')

# It contains 150 Validation samples
Data_val_2017    = np.zeros([val_number, height, width, channels])
Label_val_2017   = np.zeros([val_number, height, width])

print('Reading TN3K Validation')
print(Val_list)
for idx in range(len(Val_list)):
    print(idx+1)
    img = sc.imread(Val_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    Data_val_2017[idx, :,:,:] = img

    b = Val_list[idx]    
    a = b[0:30]
    b = b.split('.')[1].split('/')[5] 
    add = (a+ 'masks/' + b +'.png')    
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_val_2017[idx, :,:] = img2    
         
print('Reading ISIC 2017 Validation finished')


#############################################################################
#############################################################################
##############################Test_img_read##################################
#############################################################################
#############################################################################

Test_add = 'test/images'

Test_list = glob.glob(Dataset_add+ Test_add+'/*.png')

# It contains 600 test samples
Data_test_2017    = np.zeros([test_number, height, width, channels])
Label_test_2017   = np.zeros([test_number, height, width])

print('Reading TN3K Test')
print(Test_list)
for idx in range(len(Test_list)):
    print(idx+1)
    img = sc.imread(Test_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    Data_test_2017[idx, :,:,:] = img

    b = Test_list[idx]    
    a = b[0:31]
    b = b.split('.')[1].split('/')[5] 
    add = (a+ 'masks/' + b +'.png')    
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_test_2017[idx, :,:] = img2    
         
print('Reading TN3K Test finished')

################################################################ Make the train and test sets ########################################    

Train_img      = Data_train_2017
Validation_img = Data_val_2017
Test_img       = Data_test_2017

Train_mask      = Label_train_2017
Validation_mask = Label_val_2017
Test_mask       = Label_test_2017


np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)
