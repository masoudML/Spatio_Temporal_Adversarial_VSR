from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rbpn import Net as RBPN 
from rbpn import FeatureExtractor, Discriminator
from data import get_training_set, get_eval_set, get_test_set
#from tensorboardX import SummaryWriter
#writer = SummaryWriter()
import torchvision
import torchvision.models as models
import math
import numpy as np

import pdb
import socket
import time
import PWCNet
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import cv2
from scipy.ndimage import imread
from math import ceil
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./vimeo_septuplet/sequences')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--other_dataset', type=bool, default=False, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='3x_dl10VDBPNF7_epoch_84.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='F7', help='Location to save checkpoint models')
parser.add_argument('--pretrained_disc', default='', help='sr pretrained DISC base model')
parser.add_argument('--chop_forward', type=bool, default=False)


opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

def train(epoch):
    epoch_loss = 0
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0    
    model.train()
    iteration = 0
    avg_psnr = 0.0
    counter = 0.0
    for iteration, batch in enumerate(training_data_loader, 1):
        #input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]
        disc_flow, disc_neigbor, input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

'''
        for i in range(1): #range(len(neigbor)):
            n = neigbor[i]
            in_img = transforms.ToPILImage()(input[0])
            in_img.save("./in_img.png","PNG")
            neighbor_img = transforms.ToPILImage()(n[0])
            neighbor_img.save("./neighbor_img.png","PNG")
            print('max input')
            print(torch.max(input))
            print('#################################################')
            flow_neighbor_pwc = get_pwc_flow(pwc_flow, "./in_img.png","./neighbor_img.png")
#            flow_neighbor_pwc = 20 * nn.Upsample(scale_factor=4, mode='bilinear')(flow_neighbor_pwc)
            #flow_input = torch.cat((input, n), dim=1)
            #flow_neighbor_pwc = pwc_flow(flow_input)
            print('----- PWC flow ---- ')
            print(flow_neighbor_pwc.shape)
            print(flow_neighbor_pwc)
            print(' ----- pyflow flow --')
            print(flow[i].shape)
            print(flow[i])
            
#            print('flow relative error: ', rel_error(flow_neighbor_pwc.cpu().detach().numpy(), flow[i].cpu().detach().numpy()))

            objectOutput = open('./pwc_flow.flo', 'wb')
            tens = flow_neighbor_pwc[0]
            #tens = flow[i][0]
            np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objectOutput)
            np.array([tens.size(2), tens.size(1)], np.int32).tofile(objectOutput)
            np.array(tens.cpu().detach().numpy().transpose(1, 2, 0), np.float32).tofile(objectOutput)
            objectOutput.close()
            
            objectOutput = open('./pyflow_flow.flo', 'wb')
            #tens = flow_neighbor_pwc[0]
            tens = flow[i][0]
            np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objectOutput)
            np.array([tens.size(2), tens.size(1)], np.int32).tofile(objectOutput)
            np.array(tens.cpu().detach().numpy().transpose(1, 2, 0), np.float32).tofile(objectOutput)
            objectOutput.close()

            print(ff.shape)
'''
        if cuda:
            input = Variable(input).cuda(gpus_list[0])
            target = Variable(target).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input, neigbor, flow)
        
        if opt.residual:
            prediction = prediction + bicubic


        target_left = disc_neigbor[2]
        target_right = disc_neigbor[3]

        #Discriminator Neigbor Images
        r_neigbor = []
        r_neigbor.append(target_left.cuda(1))
        r_neigbor.append(target_right.cuda(1))        
        if cuda:
            #High Resolution Real and Fake assignment for discriminator
            high_res_real = Variable(target).cuda(1)
            high_res_fake = prediction.cuda(1)
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).cuda(1)
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).cuda(1)
        else:
            #High Resolution Real and Fake assignment for discriminator
            high_res_real = Variable(target)
            high_res_fake = prediction
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7)
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3)

        #Discriminator Flow Images
        dflow = []
        dflow.append(disc_flow[2].cuda(1))
        dflow.append(disc_flow[3].cuda(1))

        discriminator_loss = adversarial_criterion(discriminator(high_res_real,r_neigbor,dflow), target_real) + \
                             adversarial_criterion(discriminator(Variable(high_res_fake.data).cuda(1),r_neigbor,dflow), target_fake)
        generator_adv_loss = 1e-3*adversarial_criterion(discriminator(Variable(high_res_fake.data).cuda(1),r_neigbor,dflow), ones_const)

        target_norm = (target/torch.max(target)).to('cuda:1')
        prediction_norm = (prediction/torch.max(prediction)).to('cuda:1')

        real_features = Variable(feature_extractor(target_norm).data).cuda(1)
        fake_features = feature_extractor(prediction_norm)
        
        vgg_loss = 0.006*content_criterion(fake_features, real_features)

        loss = criterion(high_res_fake, high_res_real) + vgg_loss + generator_adv_loss
        t1 = time.time()

        #--- VGG/Context Loss
        mean_generator_content_loss += vgg_loss.data

        #--- Gen Av Loss
        mean_generator_adversarial_loss += generator_adv_loss.data #  generator_adv_loss

        #--- Disc Adv Loss
        mean_discriminator_loss += discriminator_loss.data


        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        discriminator_loss.backward()
        optim_discriminator.step()

        avg_psnr1 = PSNR(target[0], prediction[0])
        avg_psnr2 = PSNR(target[1], prediction[1])
        avg_psnr += (avg_psnr1+avg_psnr2)/2.0
        counter +=1

        #if iteration % 10 ==0:
        #    avg_psnr = eval(model)
        #    if avg_psnr > best_psnr:
        #        print('Save Best Model PSNR: '+ str(avg_psnr))
        #        best_psnr = avg_psnr
        #        checkpoint(epoch)
        #    model.train()

        
        #print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data[0], x input')(t1 - t0)))
        print('#########################################################################')
        print("===> Epoch[{}]({}/{}): PSNR: {:.4f},Loss: {:.4f}, VGG Loss {:.8f}, genAdv Loss {:.8f}, discAdv Loss {:.8f}  || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader),(avg_psnr1+avg_psnr2)/2.0,loss.data,vgg_loss, generator_adv_loss,discriminator_loss,  (t1 - t0)))
#    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f} Avg. Disc Loss {:.8f}".format(epoch, epoch_loss / len(training_data_loader),mean_discriminator_loss/len(training_data_loader)))
 #   writer.add_scalar('data/avg_epoch_loss', epoch_loss/len(training_data_loader), epoch)
 #   writer.add_scalar('data/avg_gen_vgg_loss', mean_generator_content_loss/len(training_data_loader), epoch)
 #   writer.add_scalar('data/avg_gen_adv_loss', mean_generator_adversarial_loss/len(training_data_loader), epoch)
 #   writer.add_scalar('data/avg_disc_adv_loss', mean_discriminator_loss/len(training_data_loader), epoch)

    return (avg_psnr/counter)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+str(opt.upscale_factor)+'x_gen_'+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    disc_model_out_path = opt.save_folder+str(opt.upscale_factor)+'x_disc_'+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    torch.save(discriminator.state_dict(), disc_model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))



def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def get_pwc_flow(model, im1_fn,im2_fn):
    im_all = [imread(img) for img in [im1_fn, im2_fn]]
    im_all = [im[:, :, :3] for im in im_all]
    # rescale the image size to be multiples of 64
    divisor = 64
    H = im_all[0].shape[0]
    W = im_all[0].shape[1]

    H_ = int(ceil(H/divisor) * divisor)
    W_ = int(ceil(W/divisor) * divisor)
    for i in range(len(im_all)):
       im_all[i] = cv2.resize(im_all[i], (W_, H_))
  #  print(im1.shape)
  #  print(im2.shape)

    for _i, _inputs in enumerate(im_all):
       im_all[_i] = im_all[_i][:, :, ::-1]
       im_all[_i] = 1.0 * im_all[_i]/255.0
       im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
       im_all[_i] = torch.from_numpy(im_all[_i])
       im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])    
       im_all[_i] = im_all[_i].float()

    flow_input = torch.autograd.Variable(torch.cat(im_all,1), volatile=True).cuda(0)
  #  im1 = 1.0 * im1/255.0
  #  im1 = 1.0 * im1/255.0
 #   flow_input = torch.cat((im1,im2), dim=1)
    #flow_input = flow_input.reshape(1,flow_input.shape[0],flow_input.shape[1], flow_input.shape[2])
    #print(flow_input.shape)
    #print(flow_input.type())
    
    #flow_input = flow_input.cuda(0)
    
    #flow_input = torch.cuda.FloatTensor(flow_input.numpy())#.cuda(0)
    #flow_input = Variable(flow_input, volatile=True).cuda(0)
    model.eval()
    flow_neighbor_pwc = model(flow_input)
    
    if True:
       return flow_neighbor_pwc   
 
    flow_neighbor_pwc = flow_neighbor_pwc[0] * 20.0
    flow_neighbor_pwc = flow_neighbor_pwc.cpu().data.numpy()
    print('**************************************************')
    print(flow_neighbor_pwc.shape)
    # scale the flow back to the input size
    #flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
    u_ = cv2.resize(flow_neighbor_pwc[0,:,:],(64,64))
    v_ = cv2.resize(flow_neighbor_pwc[1,:,:],(64,64))
    flow_neighbor_pwc = np.dstack((u_,v_)) 
    flow_neighbor_pwc = np.transpose(flow_neighbor_pwc, (2, 0, 1))    
    flow_neighbor_pwc = flow_neighbor_pwc.reshape((1, flow_neighbor_pwc.shape[0], flow_neighbor_pwc.shape[1],flow_neighbor_pwc.shape[2]))
    flow_neighbor_pwc = torch.from_numpy(flow_neighbor_pwc).cuda(0)
    print('--------------------------------------------------')
    print(flow_neighbor_pwc.shape)
    print('**************************************************')
    return flow_neighbor_pwc


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    gt = gt[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    imdff = pred - gt
    rmse = torch.sqrt(torch.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def eval(model,predicted, target):
    model.eval()
    count=1
    avg_psnr_predicted = 0.0
    for batch in testing_data_loader:
        input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]

        with torch.no_grad():
            input = Variable(input).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, neigbor, flow, model, opt.upscale_factor)
        else:
            with torch.no_grad():
                prediction = model(input, neigbor, flow)

        if opt.residual:
            prediction = prediction + bicubic

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
       #  save_img(prediction.cpu().data, str(count), True)

        #save_img(target, str(count), False)

        #prediction=prediction.cpu()
        #prediction = prediction.data[0].numpy().astype(np.float32)
        #prediction = prediction*255.

        #target = target.squeeze().numpy().astype(np.float32)
        #target = target*255.

        #psnr_predicted = PSNR(prediction,target, shave_border=opt.upscale_factor)
        #avg_psnr_predicted += psnr_predicted
        count+=1
        target = target.cuda(0)
        psnr = PSNR(prediction, target)
        avg_psnr_predicted += psnr

    return (avg_psnr_predicted/count)

def eval_test(model):
    model.eval()
    count=1
    avg_psnr_predicted = 0.0
    for batch in testing_data_loader:
        input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]

        with torch.no_grad():
            input = Variable(input).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, neigbor, flow, model, opt.upscale_factor)
        else:
            with torch.no_grad():
                prediction = model(input, neigbor, flow)

        if opt.residual:
            prediction = prediction + bicubic

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
       #  save_img(prediction.cpu().data, str(count), True)

        #save_img(target, str(count), False)

        #prediction=prediction.cpu()
        #prediction = prediction.data[0].numpy().astype(np.float32)
        #prediction = prediction*255.

        #target = target.squeeze().numpy().astype(np.float32)
        #target = target*255.

        #psnr_predicted = PSNR(prediction,target, shave_border=opt.upscale_factor)
        #avg_psnr_predicted += psnr_predicted
        count+=1
        target = target.cuda(0)
        psnr = PSNR(prediction, target)
        avg_psnr_predicted += psnr

    return (avg_psnr_predicted/count)



cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.file_list, opt.other_dataset, opt.patch_size, opt.future_frame)
#test_set = get_eval_set(opt.test_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
#testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

print('===> Loading datasets')
test_set = get_test_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.file_list, opt.other_dataset, opt.future_frame)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)


print('===> Building model ', opt.model_type)
if opt.model_type == 'RBPN':
    model = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=opt.nFrames, scale_factor=opt.upscale_factor) 
    discriminator = Discriminator().cuda(1) #1


feature_extractor = FeatureExtractor(models.vgg19(pretrained=True).cuda(1))
#print(feature_extractor)
model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()
ones_const = Variable(torch.ones(opt.batchSize, 1).cuda(1))

pwc_flow = PWCNet.__dict__['pwc_dc_net']("./PWCNet/pwc_net.pth.tar").cuda(0)

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    dmodel_name = os.path.join(opt.pretrained_disc)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')
    if os.path.exists(dmodel_name):
        print('RESTORE: DISCRIMINATOR Path does exist!')
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        discriminator.load_state_dict(torch.load(dmodel_name, map_location=lambda storage, loc: storage),strict=False)
        print('Pre-trained Disc model is loaded.')


if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.lr)

for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    avg_psnr = train(epoch)
    print('#########################################################################')
    print('Epoch avg PSNR :' + str(avg_psnr))
    #test()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)

# export scalar data to JSON for external processing
#writer.export_scalars_to_json("./all_scalars.json")
#writer.close()
