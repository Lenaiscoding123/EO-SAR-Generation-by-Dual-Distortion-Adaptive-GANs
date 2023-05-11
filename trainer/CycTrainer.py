# !/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR,Logger,ReplayBuffer
from .utils import weights_init_normal,get_config
from .datasets import ImageDataset,ImageDataset_trio,ValDataset,ValDataset_gen
from Model.CycleGan import *
from .utils import Resize,smooothing_loss
from .utils import Logger
from .reg import Reg, Reg_trio
from torchvision.transforms import RandomAffine
from torchvision.transforms import RandomAffine,ToPILImage
from .transformer import Transformer_2D
from skimage import measure 
import numpy as np
import cv2
from PIL import Image
import seaborn
# import cpbd

class MyCropTransform(object):
    """Rotate by one of the given angles."""

    def __init__(self, H, W):
        self.h = H
        self.w = W

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return transforms.functional.crop(img,top=0,left=0,height=self.h,width=self.w)
 



class Cyc_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netG_B2A = Generator(config['input_nc'], config['output_nc']).cuda()
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))


        self.netD_B = Discriminator(config['input_nc']).cuda()
        self.netD_A = Discriminator(config['input_nc']).cuda()

        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        self.R_A = Reg(config['size'], config['size'],config['input_nc'],config['input_nc']).cuda()
        self.R_B = Reg(config['size'], config['size'],config['input_nc'],config['input_nc']).cuda()

        self.spatial_transform = Transformer_2D().cuda()

        self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_R_B = torch.optim.Adam(self.R_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
            

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])

        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        #Dataset loader
        transforms_1 = [transforms.Resize((config['size'], config['size'])),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                        
    
        transforms_2 = [transforms.Resize((config['size'], config['size'])),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]


        
        self.dataloader = DataLoader(ImageDataset(config['dataroot'], transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False),
                            batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])

        val_transforms = [transforms.Resize((config['size'], config['size'])),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        gen_transforms = [MyCropTransform(config['size'], config['size']),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        
        
        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_ =val_transforms, unaligned=False),
                            batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

        # if config['gen_test']:
        #     dataset = ValDataset_gen(config['gen_dataroot'], transforms_ =val_transforms, unaligned=False)
        #     self.val_data = DataLoader(dataset,
        #                         batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])
 
       # Loss plot
        self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))       
        
    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
              
                self.optimizer_R_A.zero_grad()
                self.optimizer_R_B.zero_grad()
                self.optimizer_G.zero_grad()
                
                fake_B = self.netG_A2B(real_A)
                pred_fake = self.netD_B(fake_B)
                loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                
                
                fake_A = self.netG_B2A(real_B)
                pred_fake = self.netD_A(fake_A)
                loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                
                
                Trans = self.R_A(fake_B,real_B) 
                SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                
            
                Trans_ = self.R_B(fake_A,real_A) 
                SysRegist_B2A = self.spatial_transform(fake_A,Trans_)
                SR_loss_ = self.config['Corr_lamda'] * self.L1_loss(SysRegist_B2A,real_A)###SR
                SM_loss_ = self.config['Smooth_lamda'] * smooothing_loss(Trans_)

          

                


               
                recovered_A = self.netG_B2A(fake_B)
                recovered_B = self.netG_A2B(fake_A)

                
                
                loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                # Total loss
         
                loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss + SM_loss + SR_loss_ + SM_loss_ 
                            
                
                loss_Total.backward()
                self.optimizer_G.step()
                self.optimizer_R_A.step()
                self.optimizer_R_B.step()
                
                ###### Discriminator A ######
                self.optimizer_D_A.zero_grad()
                # Real loss
                pred_real = self.netD_A(real_A)
                loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                # Fake loss
                fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                pred_fake = self.netD_A(fake_A.detach())
                loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake)
                loss_D_A.backward()

                self.optimizer_D_A.step()
                ###################################

                ###### Discriminator B ######
                self.optimizer_D_B.zero_grad()

                # Real loss
                pred_real = self.netD_B(real_B)
                loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                # Fake loss
                fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                pred_fake = self.netD_B(fake_B.detach())
                loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)
                loss_D_B.backward()

                self.optimizer_D_B.step()
                ###################################    
            
                self.logger.log({'loss_D_B': loss_D_B,},
                    images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})#,'SR':SysRegist_A2B

            # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            torch.save(self.netG_B2A.state_dict(), self.config['save_root'] + 'netG_B2A.pth')
            
            torch.save(self.R_A.state_dict(), self.config['save_root'] + 'Regist.pth')
            torch.save(self.R_B.state_dict(), self.config['save_root'] + 'Regist_B.pth')
            #torch.save(netD_A.state_dict(), 'output/netD_A_3D.pth')
            #torch.save(netD_B.state_dict(), 'output/netD_B_3D.pth')
                                  

    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        #self.R_A.load_state_dict(torch.load(self.config['save_root'] + 'Regist.pth'))
        with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                BLUR = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    path = batch['BASE_PATH'][0]
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                   
                    fake_B = self.netG_A2B(real_A)
                    
                    fake_B = fake_B.detach().cpu().numpy().squeeze()
                    real_A = real_A.detach().cpu().numpy().squeeze()


                    real_B_ssim = np.transpose(real_B, (1, 2, 0))
                    real_B_ssim = real_B_ssim[:,:,0]
                    fake_B_ssim = np.transpose(fake_B, (1, 2, 0))
                    fake_B_ssim = fake_B_ssim[:,:,0]


                    mae = self.MAE(fake_B,real_B)
                    psnr = self.PSNR(fake_B,real_B)
                    ssim = measure.compare_ssim(fake_B_ssim,real_B_ssim,multichannel=True)
                    
                    MAE += mae
                    PSNR += psnr
                    SSIM += ssim 
                    num += 1

                    real_B_numpy = (np.transpose(real_B, (1, 2, 0)) + 1) / 2.0 * 255.0
                    real_A_numpy = (np.transpose(real_A, (1, 2, 0)) + 1) / 2.0 * 255.0
                    image_numpy = (np.transpose(fake_B, (1, 2, 0)) + 1) / 2.0 * 255.0
                    real_B_pil = Image.fromarray(real_B_numpy.astype(np.uint8))  
                    real_A_pil = Image.fromarray(real_A_numpy.astype(np.uint8))
                    image_pil = Image.fromarray(image_numpy.astype(np.uint8))                                             


                    if not os.path.exists(self.config["image_save"]):
                        os.makedirs(self.config["image_save"])
                    image_pil.save(self.config['image_save']+path[:-4]+'_fake.png')
                    real_B_pil.save(self.config['image_save']+path[:-4]+'_real.png')
                    real_A_pil.save(self.config['image_save']+path[:-4]+'_realA.png')
                print ('MAE:',MAE/num)
                print ('PSNR:',PSNR/num)
                print ('SSIM:',SSIM/num)


    def test_DOTA(self,):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        #self.R_A.load_state_dict(torch.load(self.config['save_root'] + 'Regist.pth'))
        with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                BLUR = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    path = batch['BASE_PATH']
                    
                    fake_B = self.netG_A2B(real_A)
                    
                    fake_B = fake_B.detach().cpu().numpy().squeeze()
                    real_A = real_A.detach().cpu().numpy().squeeze()

                    
                    image_numpy = (np.transpose(fake_B, (1, 2, 0)) + 1) / 2.0 * 255.0
                    
                    image_pil = Image.fromarray(image_numpy.astype(np.uint8)) 

                    if not os.path.exists(self.config["image_save"]):
                        os.makedirs(self.config["image_save"])                                            
                    
                    image_pil.save(self.config['image_save']+path)
            
        
 
                    
       


    def PSNR(self,fake,real):
  

       mse = np.mean(((fake+1)/2. - (real+1)/2.) ** 2 )
       if mse < 1.0e-10:
          return 100
       else:
           PIXEL_MAX = 1
           return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
            
            
    def MAE(self,fake,real):
       
        mae = np.abs(fake-real).mean()
        return mae/2     #from (-1,1) normaliz  to (0,1)
            



 