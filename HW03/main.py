import os





# cmds=[
# 'python train.py --model_name vgg16  --img_size 224 --batch_size 16 --num_epochs 500 --logger',
# 'python train.py --model_name vgg16  --img_size 224 --batch_size 16 --num_epochs 500 --use_unlabeled --logger',
# 'python train.py --model_name tiny_cnn --img_size 128 --batch_size 128 --num_epochs 500 --logger',
# 'python train.py --model_name tiny_cnn --img_size 128 --batch_size 128 --num_epochs 500 --use_unlabeled --logger',
# 'python train.py --model_name cnn --img_size 256 --batch_size 16 --num_epochs 500 --use_unlabeled --logger',
# ]
#
# for cmd in cmds:
#     print(cmd)
#     os.system(cmd)

# cmd='python train2.py --model_name vgg16  --img_size 224 --batch_size 16 --num_epochs 500 --use_unlabeled --pretrained --threshold 0.85 --lr 0.00005'
# cmd='python train2.py --model_name cnn  --img_size 256 --batch_size 16 --num_epochs 500 --use_unlabeled --threshold 0.75'
# cmd='python train2.py --model_name tiny_cnn  --img_size 128 --batch_size 128 --num_epochs 500 --use_unlabeled --threshold 0.75'

cmd='python train2.py --model_name resnet18  --img_size 256 --batch_size 64 --num_epochs 500 --use_unlabeled --threshold 0.85 --lr 0.00005'
os.system(cmd)