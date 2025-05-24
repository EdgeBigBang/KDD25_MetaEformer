
python -u run.py \
  --dataset 'Cloud' \
  --root_path ./dataset/ \
  --data CBW.npy \
  --enc_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --mpp_update 50 \
  --sim_num 70 \
  --threshold 0.15 \
  --wave_class 650 \
  --batch_size 256 \
  --freq 'd' \
  --learning_rate 1e-4 \
  --lradj 'TST' \
  --num_epoches 100 \
  --gpu 0