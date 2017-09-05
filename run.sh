# Run L-Char-LSTM 
# Use last 25% gt 
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lang_gt025
# Use all gt
python  main.py --batch-size=32 --noImg --model=lang --gt-range=1. --save-dir=./save_lang_gt100

# Run L-Char-LSTM using different text_widnow size
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
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25 --save-dir=./save_lms_lang_gt025_tw210 --text-window=210 --train_annFile=./EMNLP17_Twitch_LOL/lms_train.txt --val_annFile=./EMNLP17_Twitch_LOL/lms_val.txt

python  main.py --batch-size=32 --noImg --model=lang --gt-range=1.0 --save-dir=./save_lms_lang_gt100_tw210 --text-window=210 --train_annFile=./EMNLP17_Twitch_LOL/lms_train.txt --val_annFile=./EMNLP17_Twitch_LOL/lms_val.txt

# NALCS final test ----------------------------------------------------------------------
# Language model
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25  --text-window=210  --train_annFile=./EMNLP17_Twitch_LOL/nalcs_trainval.txt  --val_annFile=./EMNLP17_Twitch_LOL/nalcs_test.txt --save-dir=./save_lang_test

# Vision Model
python main.py --batch-size=32 --model=vision  --preTrained  --gt-range=0.25 --multi-frame=16 --train_annFile=./EMNLP17_Twitch_LOL/nalcs_trainval.txt  --val_annFile=./EMNLP17_Twitch_LOL/nalcs_test.txt --save-dir=./save_LSTMCNN_test --workers=12

# Multi  Model
python main.py --batch-size=32 --model=multi  --preTrained  --gt-range=0.25  --text-window=210 --multi-frame=16 --train_annFile=./EMNLP17_Twitch_LOL/nalcs_trainval.txt  --val_annFile=./EMNLP17_Twitch_LOL/nalcs_test.txt --save-dir=./save_Multi_test --workers=12

# LMS final test ----------------------------------------------------------------------
# Language model
python  main.py --batch-size=32 --noImg --model=lang --gt-range=0.25  --text-window=210  --train_annFile=./EMNLP17_Twitch_LOL/lms_trainval.txt  --val_annFile=./EMNLP17_Twitch_LOL/lms_test.txt --save-dir=./save_lang_test_lms

# Vision Model
python main.py --batch-size=32 --model=vision  --preTrained  --gt-range=0.25 --multi-frame=16 --train_annFile=./EMNLP17_Twitch_LOL/lms_trainval.txt  --val_annFile=./EMNLP17_Twitch_LOL/lms_test.txt --save-dir=./save_LSTMCNN_test_lms --workers=12

# Multi  Model
python main.py --batch-size=32 --model=multi  --preTrained  --gt-range=0.25  --text-window=210 --multi-frame=16 --train_annFile=./EMNLP17_Twitch_LOL/lms_trainval.txt  --val_annFile=./EMNLP17_Twitch_LOL/lms_test.txt --save-dir=./save_Multi_test_lms --workers=12

