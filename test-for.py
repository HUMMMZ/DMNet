import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

from models.LightMUNet import LightMUNet
from models.VMUNet.vmunet import VMUNet
from models.HNet import H_Net_137
from models.HMamba.MHNet import H_Net
from models.MDNet.DMNet import DMNet
from models.fcn import FCN8s
from models.unet import Unet
from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from models.trfe import TRFENet
from models.RN_Net import RNNet
from models.HMamba_skip.MHNet import H_Net_skip

from engine import *
import os
import sys
from utils import cal_params_flops
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


# 计算模型的参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main(config):

    print('#----------Creating logger----------#')    
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    
    ##############reading checkpoint ##############
    img_names = [i for i in os.listdir(checkpoint_dir) if i.endswith("best.pth")]
    print(len(img_names))
    #img_names = img_names[1:]
    print(len(img_names))
    print(img_names)
    for i in img_names:
        print(i)        
        resume_model = os.path.join(checkpoint_dir, i)
        outputs = os.path.join(config.work_dir, 'outputs')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(outputs):
            os.makedirs(outputs)

        global logger
        logger = get_logger('test', log_dir)

        log_config_info(config, logger)





        print('#----------GPU init----------#')
        set_seed(config.seed)
        gpu_ids = [0]# [0, 1, 2, 3]
        torch.cuda.empty_cache()
    


        print('#----------Prepareing Models----------#')
        model_name = config.network 
        print('#-----------------MDNet---------------#')
        if model_name == 'UltraLight_VM_UNet':
            model_cfg = config.model_config
            model = UltraLight_VM_UNet(num_classes=model_cfg['num_classes'], 
                                   input_channels=model_cfg['input_channels'], 
                                   c_list=model_cfg['c_list'], 
                                   split_att=model_cfg['split_att'], 
                                   bridge=model_cfg['bridge'],)
        elif model_name == 'VMUNet':
            model_cfg = config.model_config
            model = VMUNet(num_classes=model_cfg['num_classes'], 
                               input_channels=model_cfg['input_channels'], 
                               depths=model_cfg['depths'],
                               depths_decoder=model_cfg['depths_decoder'],
                               drop_path_rate=model_cfg['drop_path_rate'],
                               load_ckpt_path=model_cfg['load_ckpt_path'],)
        elif model_name == 'LightMUNet':
            model = LightMUNet(out_channels=2, in_channels=model_cfg['input_channels'], init_filters=32)
        elif model_name == 'fcn':
            model = FCN8s(1)
        elif model_name == 'HNet':
            model = H_Net_137(3,1)
        elif model_name == 'DMNet':
            model = DMNet(3,1)
        elif model_name == 'HNet2':
            model = H_Net_1(3,1)
        elif model_name == 'unet':
            model = Unet(in_ch=3, out_ch=1)
        elif model_name == 'trfe':
            model = TRFENet(in_ch=3, out_ch=1)
        elif model_name == 'RNNet':
            model = RNNet(in_channels=3, num_classes=1, base_c=64)
        elif model_name == 'HNetskip':
            model = H_Net_skip(3,1)
            
        #cal_params_flops(model, 256, logger)
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
        #print(f'Model has {count_parameters(model)} parameters.')
        
        print('#----------Preparing dataset----------#')
        test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
        test_loader = DataLoader(test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True, 
                                    num_workers=config.num_workers,
                                    drop_last=True)

        print('#----------Prepareing loss, opt, sch and amp----------#')
        criterion = config.criterion
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, optimizer)
        scaler = GradScaler()


        print('#----------Set other params----------#')
        min_loss = 999
        start_epoch = 1
        min_epoch = 1


        print('#----------Testing----------#')
        best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(best_weight)
        loss = test_one_epoch(
                test_loader,
                model,
                criterion,
                logger,
                config,
            )


if __name__ == '__main__':
    config = setting_config
    main(config)
    


