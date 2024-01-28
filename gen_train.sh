python generate_data.py --name bouncing_ball_data_convex_shape_colour_test_T=300 --ntrain 1 --ntest 1000 --datasets-root /data2/users/cb221
#CUDA_VISIBLE_DEVICES=2 python train_VRNN.py --name VRNN_missing --train_root /data2/users/hbz15/2_body_black_white_real/train_missing --epochs 100 -b 32 --beta 1 --visual --hidden_dim 256 --latent_dim 4 --missing

#CUDA_VISIBLE_DEVICES=1 python train_VrSLDS.py --name nascar_5_lstm_sequential --experiment nascar --lr 5e-4 -b 16 --seq_len 100
#CUDA_VISIBLE_DEVICES=0 python train_KVAE.py --name KVAE_rnn --beta 0.3 --train_root /data2/users/cb221/bouncing_ball_data_square_black_white_small/train/ -b 32 --seq_len 20 --kf_steps 10


#CUDA_VISIBLE_DEVICES=0 python train_KVAE.py --name KVAE_glow_2_deconv_emission_2x2 --train_root /data2/users/cb221/bouncing_ball_data_square_black_white_small_16/train -b 40 --seq_len 20 --lr 7e-3 --model greparam