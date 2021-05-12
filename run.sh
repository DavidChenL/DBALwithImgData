CUDA_VISIBLE_DEVICES=2 python3 main.py --batch_size 128 \
                  --epochs 50 \
                  --lr 1e-3 \
                  --seed 369 \
                  --experiments 3 \
                  --dropout_iter 100 \
                  --query 10 \
                  --acq_func 0 \
                  --val_size 100 \
                  --result_dir result_npy