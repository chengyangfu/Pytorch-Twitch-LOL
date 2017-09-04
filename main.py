'''
Author : Cheng-Yang Fu <cyfu@cs.unc.edu> 
'''
import argparse
import os
import shutil
import time
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms
torch.backends.cudnn.benchmark = True

from twitch_data_loader import *
from models import *

parser = argparse.ArgumentParser(description='PyTorch Video Summary')
parser.add_argument('--train_data_path', dest='train_data_path',
                    help='Directory contains the training images',
                    default='/net/bvisionserver3/playpen10/cyfu/twitch_lol/',
                    type=str, metavar='PATH')

parser.add_argument('--train_annFile', dest='train_ann',
                    help='List file contains location of images and labels',
                    default='/net/bvisionserver3/playpen10/cyfu/twitch_lol/nalcs_train.txt',
                    type=str, metavar='PATH')

parser.add_argument('--val_annFile', dest='val_ann',
                    help='List file contains location of images and labels',
                    default='/net/bvisionserver3/playpen10/cyfu/twitch_lol/nalcs_val.txt',
                    type=str, metavar='PATH')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--visualize', dest='visualize', action='store_true',
                    help='Visualize the prediction')
parser.add_argument('--threshold', default=0.6, type=float,
                    help='threshold for visualization')
parser.add_argument('--save-dir', dest='save_dir',
                    default='save_models', type=str,
                    help='The directory used to save the trained models')
parser.add_argument('--validate', dest='validate', action='store_true', default=False,
                    help='Run Validation during training')
parser.add_argument('--noImg', dest='noImg', action='store_true', default=False,
                    help='Not using Images')
parser.add_argument('--preTrained', dest='preTrained', action='store_true', default=False,
                    help='Use preTrained model for vision model')
parser.add_argument('--multi-frame', default=1, type=int, metavar='N',
                    help='Multi Frame (for LSTM on CNN) (default: 1)')
parser.add_argument('--model', dest='model', help='vision, lang, multi', default='lang', type=str)
parser.add_argument('--gt-range', default=0.25, type=float,
                    metavar='N', help='how much gt_range we used for the training')
parser.add_argument('--text-window', default=150, type=int,
                    metavar='N', help='text window size')
parser.add_argument('--word', dest='word', action='store_true', default=False,
                    help='Use Word-level LSTM')

best_mAP  = 0

def main():
    global args, best_mAP
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Print all the setting first
    print '=> Setting:'
    for it_item in vars(parser.parse_args()).iteritems():
        print it_item

    # Create DataLoader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # if args.evaluate  == False:
    print "=> Loading Training Data: "
    trainLoader = Twitch(root=args.train_data_path, list_file=args.train_ann,
                         transform=transforms.Compose([
                             # Because all video are resized to 224x224 first
                             # to save the time.
                             # transforms.Scale((224,224)),
                             transforms.ToTensor(),
                             normalize,]),
                         prod_Img=not(args.noImg), multi_frame=args.multi_frame,
                         text_window=args.text_window, gt_range=args.gt_range, word=args.word)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=trainLoader.WeightedSampling.tolist(), num_samples=10000)

    train_loader = torch.utils.data.DataLoader(trainLoader,
                                               batch_size=args.batch_size, 
                                               num_workers=args.workers, pin_memory=True,
                                               sampler=sampler)

    #text_model = CharModel(100, 128, 2, output_size=2, rnntype='RNN')
    if args.model == 'vision':
        model = VisionModel(preTrained=args.preTrained)
    elif args.model == 'lang':
        if args.word:
            model = LangModel(preTrained=args.preTrained, input=len(trainLoader.corpus))
        else:
            model = LangModel(preTrained=args.preTrained)
    elif args.model == 'multi':
        model = MultiModel(preTrained=args.preTrained)
    else:
        print 'args.model = {} is not supported'.format(args.model)
        sys.exit('')

    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_pec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss().cuda()


    if args.evaluate :
        print("=> Loading Evaluation Data: ")
        valLoader = Twitch(root=args.train_data_path, list_file=args.val_ann,
            transform=transforms.Compose([
#                transforms.Scale((224,224)),
                transforms.ToTensor(),
                normalize,
                ]), prod_Img = not (args.noImg), multi_frame=args.multi_frame, text_window=args.text_window,
            word=args.word, corpus = trainLoader.corpus)

        sampler =  SampleSequentialSampler(valLoader, 30)

        val_loader = torch.utils.data.DataLoader( valLoader,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,  pin_memory=True, sampler= sampler)

        val(val_loader, model)

        sys.exit()

    # Number of Epochs
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, optimizer, criterion, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'lr': lr,
            'state_dict': model.state_dict(),
        }, True, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

def train(train_loader, model, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()


    model.train()
    end = time.time()

    for it, (img, text, label) in enumerate(train_loader):

        if not args.noImg:
            img_var = Variable(img)
        else:
            img_var = []

        label = label
        label_var = Variable(label).cuda()

        if args.word :
            text = text_util.word_linesToTensor(text, train_loader.dataset.corpus)
        else:
            text = text_util.linesToTensor(text)
        text_var = Variable(text)

        if args.model == 'vision' :
            output = model(img_var)
        elif args.model == 'lang' :
            output = model(text_var)
        elif args.model == 'multi' :
            output = model(img_var, text_var)

#        output = model(img_var, text_var)
        loss = criterion(output, label_var)


        # measure accuracy and record loss
        prec1 = accuracy(output, label_var)[0]
        losses.update(loss.data[0], label.size(0))
        top1.update(prec1.data[0], label.size(0))



        # compute gradient and do SGD step
        optimizer.zero_grad()

        # backward
        loss.backward()
        clip_gradient(model, 10.)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if it % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time:{batch_time.val:.1f}({batch_time.avg:.1f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, it, len(train_loader), batch_time=batch_time,
                   loss=losses, top1=top1))

def val(val_loader, model):
    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()


    model.eval()
    end = time.time()

    pred_sum = 1
    gt_sum = 1
    correct_sum = 0

    for it, (img, text, label) in enumerate(val_loader):

        if not args.noImg:
            img_var = Variable(img)
        else:
            img_var = []

        label = label
        label_var = Variable(label).cuda()


        if args.word :
            text = text_util.word_linesToTensor(text, val_loader.dataset.corpus)
        else:
            text = text_util.linesToTensor(text)
        text_var = Variable(text)


        if args.model == 'vision' :
            output = model(img_var)
        elif args.model == 'lang' :
            output = model(text_var)
        elif args.model == 'multi' :
            output = model(img_var, text_var)

        correct_len, pred_len, gt_len = fmeasure(output, label_var)
        correct_sum += correct_len
        pred_sum += pred_len
        gt_sum += gt_len




        batch_time.update(time.time() - end)
        end = time.time()

        if it > 1000:

            if it % args.print_freq == 0 and pred_sum >0 and gt_sum >0:
                precision = correct_sum / float(pred_sum)
                recall = correct_sum / float(gt_sum)
                f1 = (2*precision*recall / (precision + recall)) * 100

                print('[{}/{}], prec:{}, recall:{}, f1:{}'.format(it, len(val_loader), precision, recall, f1))
        else :
            if it % args.print_freq == 0:
                print('[{}/{}]'.format(it, len(val_loader)))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def fmeasure(output, target):

    _, pred = output.topk(1, 1, True, True)
    pred = pred
#    correct = pred.eq(target.view(1, -1).expand_as(pred))
    overlap = ((pred== 1) + (target == 1)).gt(1)

    overlap_len = overlap.data.long().sum()
    pred_len = pred.data.long().sum()
    gt_len   =  target.data.long().sum()

    return overlap_len, pred_len, gt_len


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for pm in model.parameters():
        if pm.requires_grad:
            #print(pm.size())
            if pm.grad is not None:
                modulenorm = pm.grad.data.norm()
                totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is not None:
                p.grad.mul_(norm)


if __name__ == '__main__':
    main()
