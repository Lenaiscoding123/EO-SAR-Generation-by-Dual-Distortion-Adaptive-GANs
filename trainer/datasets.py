import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root,count = None,transforms_1=None,transforms_2=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned
        
    def __getitem__(self, index):
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        
        A_img = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        
        #random.seed(seed)
        if self.unaligned:
            B_img = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB')
            
        
        else: 
            B_img = Image.open(self.files_B[index % len(self.files_B)]).convert('RGB') 
        item_A = self.transform1(A_img)
        item_B = self.transform2(B_img)
        
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset_trio(Dataset):
    def __init__(self, root,count = None,transforms_1=None,transforms_2=None, transforms_3=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.transform3 = transforms.Compose(transforms_3)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.files_C = sorted(glob.glob("%s/C/*" % root))
        self.unaligned = unaligned
        
    def __getitem__(self, index):
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        
        A_img = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        
        #random.seed(seed)
        if self.unaligned:
            B_img = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB')
            
        
        else: 
            B_img = Image.open(self.files_B[index % len(self.files_B)]).convert('RGB') 
            C_img = Image.open(self.files_C[index % len(self.files_C)]).convert('RGB') 
        item_A = self.transform1(A_img)
        item_B = self.transform2(B_img)
        item_C = self.transform3(C_img)
        
        return {'A': item_A, 'B': item_B, 'C': item_C}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B), len(self.files_C))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        
    def __getitem__(self, index):
        A_img = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        
        #random.seed(seed)
        if self.unaligned:
            B_img = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB')
            
        
        else: 
            B_img = Image.open(self.files_B[index % len(self.files_B)]).convert('RGB') 
        item_A = self.transform(A_img)
        item_B = self.transform(B_img)
        
        return {'A': item_A, 'B': item_B, 'BASE_PATH': os.path.basename(self.files_A[index % len(self.files_A)])}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ValDataset_gen(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        
    def __getitem__(self, index):
        A_img = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        
        
        item_A = self.transform(A_img)
        
        return {'A': item_A, 'PATH': self.files_A[index % len(self.files_A)], 'BASE_PATH': os.path.basename(self.files_A[index % len(self.files_A)])}

    def __len__(self):
        return len(self.files_A)