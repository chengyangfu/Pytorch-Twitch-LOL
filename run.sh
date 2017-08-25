python main.py --batch-size=32 --save-dir=./save_FrameCNN |& tee -a log_FrameCNN
python main.py --batch-size=32 --save-dir=./save_FrameCNN_preTrained --preTrained |& tee -a log_FrameCNN_preTrained
python main.py --batch-size=32 --save-dir=./save_LSTMCNN --multi-frame=16 --workers=8 |& tee -a log_LSTMCNN 
python main.py --batch-size=32 --save-dir=./save_LSTMCNN_preTrained --multi-frame=16 --preTrained --workers=8 |& tee -a log_LSTMCNN_preTrained 



python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lang_gt025
#python  main.py --batch-size=32 --noImg --model=lang --gt-range=1. --save-dir=./save_lang_gt100

python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lang_gt025_tw30 --text-window=30
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lang_gt025_tw60 --text-window=60
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lang_gt025_tw90 --text-window=90
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lang_gt025_tw120 --text-window=120
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lang_gt025_tw150 --text-window=150
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lang_gt025_tw180 --text-window=180
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lang_gt025_tw210 --text-window=210
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lang_gt025_tw240 --text-window=240
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lang_gt025_tw270 --text-window=270


#Word LSTM 
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lang_gt025_tw210_word --text-window=210 --word



# NALCS Vision model
python main.py --batch-size=32 --model=vision --save-dir=./save_FrameCNN_gt100 --preTrained  --gt-range=1.
python main.py --batch-size=32 --model=vision --save-dir=./save_FrameCNN_gt025 --preTrained  --gt-range=0.25
python main.py --batch-size=32 --model=vision --save-dir=./save_LSTMCNN_gt025 --preTrained  --gt-range=0.25 --multi-frame=16

# NALCS Vision(LSTM CNN) + Lang
python main.py --batch-size=32 --model=multi  --preTrained  --gt-range=0.25  --text-window=210 --multi-frame=16 --save-dir=./save_Multi_gt025


# Train lms 
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lms_lang_gt025_tw210 --text-window=210 --train_annFile=/net/bvision9/playpen/cyfu/twitch_lol_dataset/lms_train.txt --val_annFile=/net/bvision9/playpen/cyfu/twitch_lol_dataset/lms_val.txt

python  main.py --batch-size=32 --noImg --model=lang --gt-range=1.0 --save-dir=./save_lms_lang_gt100_tw210 --text-window=210 --train_annFile=/net/bvision9/playpen/cyfu/twitch_lol_dataset/lms_train.txt --val_annFile=/net/bvision9/playpen/cyfu/twitch_lol_dataset/lms_val.txt


# NALCS final test 
# Language model
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25  --text-window=210  --train_annFile=/net/bvisionserver3/playpen10/cyfu/twitch_lol/nalcs_trainval.txt  --val_annFile=/net/bvisionserver3/playpen10/cyfu/twitch_lol/nalcs_test.txt --save-dir=./save_lang_test

# Vision Model
python main.py --batch-size=32 --model=vision  --preTrained  --gt-range=0.25 --multi-frame=16 --train_annFile=/net/bvisionserver3/playpen10/cyfu/twitch_lol/nalcs_trainval.txt  --val_annFile=/net/bvisionserver3/playpen10/cyfu/twitch_lol/nalcs_test.txt --save-dir=./save_LSTMCNN_test --workers=12

# Multi  Model
python main.py --batch-size=32 --model=multi  --preTrained  --gt-range=0.25  --text-window=210 --multi-frame=16 --train_annFile=/net/bvisionserver3/playpen10/cyfu/twitch_lol/nalcs_trainval.txt  --val_annFile=/net/bvisionserver3/playpen10/cyfu/twitch_lol/nalcs_test.txt --save-dir=./save_Multi_test --workers=12


# LMS final test 
# Language model
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25  --text-window=210  --train_annFile=/net/bvisionserver3/playpen10/cyfu/twitch_lol/lms_trainval.txt  --val_annFile=/net/bvisionserver3/playpen10/cyfu/twitch_lol/lms_test.txt --save-dir=./save_lang_test_lms

# Vision Model
python main.py --batch-size=32 --model=vision  --preTrained  --gt-range=0.25 --multi-frame=16 --train_annFile=/net/bvisionserver3/playpen10/cyfu/twitch_lol/lms_trainval.txt  --val_annFile=/net/bvisionserver3/playpen10/cyfu/twitch_lol/lms_test.txt --save-dir=./save_LSTMCNN_test_lms --workers=12

# Multi  Model
python main.py --batch-size=32 --model=multi  --preTrained  --gt-range=0.25  --text-window=210 --multi-frame=16 --train_annFile=/net/bvisionserver3/playpen10/cyfu/twitch_lol/lms_trainval.txt  --val_annFile=/net/bvisionserver3/playpen10/cyfu/twitch_lol/lms_test.txt --save-dir=./save_Multi_test_lms --workers=12



