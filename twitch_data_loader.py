import torch.utils.data as data
from PIL import Image
import os
import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
import text_util
import torch
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
import copy
import torch.utils.data.sampler as sampler
from data import Dictionary 


class Twitch(data.Dataset):
    def __init__(self, root, list_file, number=1000, transform=None, text_transform=None,
            prod_Img=True, prod_Text=True, multi_frame=1, text_window=150, text_delay=0, gt_range=0.25, word=False, corpus = None):
        self.root = root
        self.__load_set(list_file)
        self.transform = transform
        self.text_transform = text_transform
        self.nums = number
        self.prod_Img = prod_Img
        self.prod_Text = prod_Text
        self.multi_frame = multi_frame
        self.video_idx = 0
        self.gt_range =  1- gt_range
        self.text_window = text_window
        self.text_delay = text_delay
        self.WeightedSampling = [] 
        self.word = word
        self.corpus = corpus 
        if self.word  and corpus == None:
            self.__set_corpus()
        counter = 0
    
        for gt in self.gt_list:
            self.WeightedSampling.extend(copy.copy(gt))

        sampling = np.array(self.WeightedSampling)
        neg_idx = np.where(sampling == 0)[0]
        pos_idx = np.where(sampling == 1)[0]
        sampling = sampling.astype(np.float32)

        begin_pos = 0 
        hl_frames = []
        for it, cur_pos in enumerate(pos_idx):
            if it+1 < len(pos_idx): 
                if((pos_idx[it+1] - cur_pos) > 1):
                    begin = int((it+1 - begin_pos) * self.gt_range) + begin_pos
                    hl_frames.extend( pos_idx[begin: it] ) 
                    begin_pos = it+1


        sampling.fill(0)
        sampling[neg_idx] = len(sampling) / float(len(neg_idx))
       # self.WeightedSampling[pos_idx] = len(self.WeightedSampling) / float(len(pos_idx))
        sampling[hl_frames] = len(sampling) / float(len(hl_frames))
        self.WeightedSampling = sampling
        
        self.sums = np.insert(np.cumsum([len(gt) for gt in self.gt_list]), 0, 0)
        print('Twitch Data Loader is ready.')

    def __load_set(self, set_file):
        with open(set_file) as f:
            lines = f.readlines()
        
        video_list= []
        text_list = []
        gt_list = []
        for line in lines:
            line = line.strip('\n')
            segs = line.split(' ')
            print('=>Load Video', segs)
            assert(len(segs) == 3)
            segs = [ os.path.join(self.root, seg) for seg in segs ]

            video_list.append(segs[0])
            cap = FFMPEG_VideoReader(segs[0])
            cap.initialize()
            #video_list.append(cap)
            print('Video: frames({})'.format(int(cap.nframes)))
            # Load text json file
            text = json.load(open(segs[1]))
            # Load GT json file
            gt   = np.load(open(segs[2]))
            print('Gt : frames({})'.format(len(gt)))
            text_list.append(text)
            gt_list.append(gt)
            

        self.video_list = video_list
        self.text_list = text_list
        self.gt_list  = gt_list

        
       
                        
    def __set_corpus(self):
        pre_dict = Dictionary()
        for lines  in self.text_list:
            for line in lines:
                if len(line) > 0:
                    words = line.split() 
                    #tokens += len(words)
                    for word in words:
                        pre_dict.add_word(word)

        pro_dict = Dictionary()
        for key in pre_dict.count: 
            if(pre_dict.count[key] > 10):
                pro_dict.add_word(key)
        self.corpus = pro_dict
                    

    

    def __getitem__(self, index):
        # Find the video first. 
        vid = np.histogram(index, self.sums)
        assert(np.sum(vid[0]) == 1)
        vid = np.where(vid[0]>0)[0][0]

        v_fmax = len(self.gt_list[vid])
        vframe = index - self.sums[vid]
        #vframes = [min(vframe + i, len(self.gt_list[vid])- 1) for i in np.arange(0, 1 +10*self.multi_frame, 10)]
        #vframes = [min(vframe, len(self.gt_list[vid])- 1)]
        #cap = self.video_list[vid]
        
        imgs = []
        if self.prod_Img:

            cap = FFMPEG_VideoReader(self.video_list[vid])
            cap.initialize()
                
            for i in range(self.multi_frame ):
                if i == 0:
                    img = cap.get_frame(vframe/cap.fps)  
                else:
                    cap.skip_frames(n=9)
                    img = cap.read_frame()
          
                img = Image.fromarray(img)
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img)    

            '''
            for v in vframes:
                #cap = cv2.VideoCapture(self.video_list[vid])
                #assert cap.isOpened() == True, 'The Video cannot be read:{}'.format(self.video_list[vid])
                

                #cap.set(1, v)
                #ret, frame= cap.read()
                img = cap.get_frame(v)
                #assert ret == True, 'Cannot Load this frame{}:{}'.format(self.video_list[vid], v)
                #cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img)    
            '''
            imgs = [img.unsqueeze(0) for img in imgs]
            imgs = torch.cat(imgs, 0)

        text = [] 
        if self.prod_Text :
            text = self.text_list[vid][min(vframe + self.text_delay, len(self.text_list[vid]))
                    : min(vframe + self.text_window + self.text_delay, len(self.text_list[vid]) )]
            text = ''.join(text)

        gt   = self.gt_list[vid][vframe]

            
        if(len(text) == 0):
            text = ' '
        #text = text_util.lineToTensor(text)
        
        return imgs, text, gt 


    def __len__(self):
        # For each epoch, we only sample from one video. 
        return len(self.WeightedSampling)



class SampleSequentialSampler(sampler.Sampler):
    """Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
        offset (int): offset between the samples
    """

    def __init__(self, data_source, offset=10):
        self.num_samples = len(data_source) 
        self.offset = offset

    def __iter__(self):
        return iter(np.arange(0, self.num_samples, self.offset ))

    def __len__(self):
        return len(np.arange(0, self.num_samples, self.offset ))


