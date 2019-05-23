import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import pyflow
from skimage import img_as_float
from random import randrange
import os.path
import PWCNet

from torch.autograd import Variable

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1, nFrames)]
    #random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]

        for i in seq:
            index = int(filepath[char_len-7:char_len-4])-i
            file_name=filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png'
            print('neighbor_filename: ' + file_name)
            print('index values are',index)    
            if os.path.exists(file_name):
                temp = modcrop(Image.open(filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png').convert('RGB'),scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath,'im'+str(nFrames)+'.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC) for j in reversed(seq)]
        disc_neigbor = [modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale) for j in reversed(seq)]
    
    return disc_neigbor, target, input, neigbor

def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames/2)
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]
        disc_neigbor = []
        seq = [x for x in range(-tt,tt+1) if x!=0]
        #random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len-7:char_len-4])+i
            file_name1=filepath[0:char_len-7]+'{0:03d}'.format(index1)+'.png'
            print('neighbor_filename: ' + file_name1)
            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame- is not exist')
                temp=input
                neigbor.append(temp)
            
    else:
        target = modcrop(Image.open(join(filepath,'im4.png')).convert('RGB'),scale)

        #print('size of target at loader is',target.size)

        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)

        print('size of INPUT at loader is',input.size)

        neigbor = []
        disc_neigbor = []
        seq = [x for x in range(4-tt,5+tt) if x!=4]
        #random.shuffle(seq) #if random sequence
        for j in seq:
            print('neighbor_filename: ' + filepath+'/im'+str(j)+'.png')
            neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC))
             
        for j in seq:
            disc_neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale))
            #im = Image.open(filepath+'/im'+str(j)+'.png').convert('RGB')
            #im.save('neig'+str(j)+'.png')
    #raise SystemExit
            #print('size of image at loader is',im.size)
    return disc_neigbor,target, input, neigbor

def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,0,1)
    return flow

def get_pwc_flow(model, im1,im2):
    im_all = [im[:, :, :3] for im in [im1,im2]]
    # rescale the image size to be multiples of 64
    #divisor = 64
    #H = im_all[0].shape[0]
    #W = im_all[0].shape[1]

    #@H_ = int(ceil(H/divisor) * divisor)
    #W_ = int(ceil(W/divisor) * divisor)
    #for i in range(len(im_all)):
    #	im_all[i] = cv2.resize(im_all[i], (W_, H_))
    print(im1.shape)
    print(im2.shape)
     
  #  for _i, _inputs in enumerate(im_all):
        #im_all[_i] = im_all[_i][:, :, ::-1]
  #      print('max - before norm')
  #      print(torch.max(im_all[_i]))
  #      im_all[_i] = 1.0 * im_all[_i]/255.0
  #      print('max - after norm')
  #      print(torch.max(im_all[_i]))
  #      print('--------')
        #im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
        #im_all[_i] = torch.from_numpy(im_all[_i])
        #im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
        #im_all[_i] = im_all[_i].float()
    
    #flow_input = torch.autograd.Variable(torch.cat(im_all,1), volatile=True).cuda(1)
    
    flow_input = torch.cat((im1,im2), dim=0)
    flow_input = flow_input.reshape(1,flow_input.shape[0],flow_input.shape[1], flow_input.shape[2])
    print(flow_input.shape)
    print(flow_input.type())
    #flow_input = flow_input.clone().cuda()
    #flow_input = torch.cuda.FloatTensor(flow_input.numpy())#.cuda(0)
    flow_input = Variable(flow_input, volatile=True).cuda(0)
    flow_neighbor_pwc = model(flow_input)
    print('----- PWC flow ---- ')
    print(flow_neighbor_pwc.shape)
    return flow_neighbor_pwc

def get_pwc_flow1(model, im1,im2):
    #im_all = [im[:, :, :3] for im in [im1,im2]]
    # rescale the image size to be multiples of 64
    #divisor = 64
    #H = im_all[0].shape[0]
    #W = im_all[0].shape[1]

    #@H_ = int(ceil(H/divisor) * divisor)
    #W_ = int(ceil(W/divisor) * divisor)
    #for i in range(len(im_all)):
    #   im_all[i] = cv2.resize(im_all[i], (W_, H_))
    #print(im1.shape)
    #print(im2.shape)

  #  for _i, _inputs in enumerate(im_all):
        #im_all[_i] = im_all[_i][:, :, ::-1]
  #      print('max - before norm')
  #      print(torch.max(im_all[_i]))
  #      im_all[_i] = 1.0 * im_all[_i]/255.0
  #      print('max - after norm')
  #      print(torch.max(im_all[_i]))
  #      print('--------')
        #im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
        #im_all[_i] = torch.from_numpy(im_all[_i])
        #im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])    
        #im_all[_i] = im_all[_i].float()

    #flow_input = torch.autograd.Variable(torch.cat(im_all,1), volatile=True).cuda(1)

    flow_input = torch.cat((im1,im2), dim=0)
    flow_input = flow_input.reshape(1,flow_input.shape[0],flow_input.shape[1], flow_input.shape[2])

    print(flow_input.shape)
    #flow_input = flow_input.reshape(1,flow_input.shape[0],flow_input.shape[1], flow_input.shape[2])
    #print(flow_input.shape)
    #print(flow_input.type())
    #flow_input = flow_input.cuda(0)
    #flow_input = torch.cuda.FloatTensor(flow_input.numpy())#.cuda(0)
    #flow_input = Variable(flow_input, volatile=True).cuda(0)

    #torch.cuda.set_device(0)
    model = model.cuda(0)
    flow_input = flow_input.cuda(0)
    model.eval()
    flow_neighbor_pwc = model(flow_input)
    print('----- get flow 0 --- PWC flow ---- ')
    print(flow_neighbor_pwc.shape)
    return flow_neighbor_pwc

def rescale_flow(x,max_range,min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo);
    iw = iw - (iw%modulo);
    img = img.crop((0, 0, ih, iw))
    return img

def get_patch(img_in, img_tar, img_nn,img_dnn, patch_size, scale, nFrames, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))#[:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))#[:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy,ix,iy + ip, ix + ip)) for j in img_nn] #[:, iy:iy + ip, ix:ix + ip]
    img_dnn = [j.crop((ty,tx,ty + tp, tx + tp)) for j in img_dnn] #[:, iy:iy + ip, ix:ix + ip]
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_nn,img_dnn, info_patch

def augment(img_in, img_tar, img_nn,img_dnn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        img_dnn = [ImageOps.flip(j) for j in img_dnn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            img_dnn = [ImageOps.mirror(j) for j in img_dnn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            img_dnn = [j.rotate(180) for j in img_dnn]
            info_aug['trans'] = True

    return img_in, img_tar, img_nn,img_dnn, info_aug
    
def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame, transform=None):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame
        pwcnet = PWCNet.__dict__['pwc_dc_net']("PWCNet/pwc_net.pth.tar")#.cuda(0)
        self.pwcnet = torch.nn.DataParallel(pwcnet)
    def __getitem__(self, index):
        if self.future_frame:
            print('############# Future Frame ###################')
            print(self.image_filenames[index])
            fn = './vimeo_septuplet/sequences/00004/0352'
            disc_neigbor, target, input, neigbor = load_img_future(fn, self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            print('############# Not Future Frames ###################')
            disc_neigbor, target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        print('### input-shape before patch  #####')
        print(index)
        print(input.size)
        print(type(input))
        if self.patch_size != 0:
            input, target, neigbor,disc_neigbor, _ = get_patch(input,target,neigbor,disc_neigbor,self.patch_size, self.upscale_factor, self.nFrames)
        
        print('### input-shape before augmentation  #####')
        print(input.size)

        if self.data_augmentation:
            input, target, neigbor,disc_neigbor, _ = augment(input, target, neigbor,disc_neigbor)
        #pwc_flow = [get_pwc_flow(self.pwcnet,input,j) for j in neigbor]
        print('### input-shape before get flow #####')
        print(input.size)
        flow = [get_flow(input,j) for j in neigbor]
        disc_flow = [get_flow(target,j) for j in disc_neigbor]
        bicubic = rescale_img(input, self.upscale_factor)

        print('before transform')
        if self.transform:
            print('after')
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            #print(input)
            neigbor = [self.transform(j) for j in neigbor]
            #pwc_flow = [get_pwc_flow1(self.pwcnet,input,j) for j in neigbor]
            disc_neigbor = [self.transform(j) for j in disc_neigbor]
            flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]
            disc_flow = [torch.from_numpy(j.transpose(2,0,1)) for j in disc_flow]

        #print('*&(&( shape of disc_neigbor',disc_neigbor[0].shape)    

        return disc_flow, disc_neigbor, input, target, neigbor, flow, bicubic

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            _, target, input, neigbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
            
        flow = [get_flow(input,j) for j in neigbor]

        bicubic = rescale_img(input, self.upscale_factor)
        
        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]
            
        return input, target, neigbor, flow, bicubic
      
    def __len__(self):
        return len(self.image_filenames)
