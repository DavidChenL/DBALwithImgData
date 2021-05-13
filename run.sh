#CUDA_VISIBLE_DEVICES=2 python3 main.py --batch_size 128 \
#                  --epochs 50 \
#                  --lr 1e-3 \
#                  --seed 369 \
#                  --experiments 3 \
#                  --dropout_iter 100 \
#                  --query 10 \
#                  --acq_func 0 \
#                  --val_size 100 \
#                  --result_dir result_npy

CUDA_VISIBLE_DEVICES=0,1,3 python3 main.py --batch_size 128 --epochs 50 --lr 1e-3 --seed 369 --experiments 3 --dropout_iter 100 --init_size 100 --query 200 --acq_func 0 --val_size 1000 --result_dir log/CIFAR10_result --dataset CIFAR10 >> log/CIFAR10_result/log.txt
#CUDA_VISIBLE_DEVICES=3 python3 main.py --batch_size 128 --epochs 50 --lr 1e-3 --seed 369 --experiments 3 --dropout_iter 100 --query 10 --acq_func 0 --val_size 1000 --result_dir log/MNIST_result --dataset MNIST >> log/MNIST_result/log.txt