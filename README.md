# pytorch-Twitch-LOL
PyTorch implementation and dataset of Video Highlight Prediction Using Audience Chat Reactions, 2017 EMNLP.


## Library Requirement 

### Apt-get install 
 
 - ffmpeg


### Python 
 - PyTorch (https://github.com/pytorch/pytorch)
 - PyTorch/Vision (https://github.com/pytorch/vision)
 - MoviePy (https://github.com/Zulko/moviepy)
 

## Dataset Download - Google Drive 

 - https://drive.google.com/drive/folders/0By9LEMeCDdboVDlHTDlqQUNHMnc?usp=sharing
 - Before training, it is important to convert encoding of video from h.264 to mpeg and resize the resolution. 
 
 ```shell
 cd EMNLP17_Twitch_LOL
 python compressVideo.py (This requires around 378 GB)
 ```
## Run Training. 

 - Check the run.sh file. This script contains all the configuration of the experiments. 
 - If you want to evaluate the trained model, simply use the same command used in training and add `-e`. 


### Citing 

Please cite this paper in your publications if it helps your research:

    @inproceedings{fu2017highlight,
      title = {Video Highlight Prediction Using Audience Chat Reactions},
      author = {Cheng-Yang Fu, Joon Lee, Mohit Bansal and Alexander C. Berg},
      booktitle = {EMNLP},
      year = {2017}
    }
