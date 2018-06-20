#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python examples/chiu.py --mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 275 --num_layers 1 \
 --char_dim 30 --num_filters 30 --tag_space 128 \
 --learning_rate 0.01 --decay_rate 0 --schedule 1 --gamma 0.0 \
 --dropout std --p_in 0 --p_rnn 0 0 --p_out 0.68 --unk_replace 0.0 --bigram \
 --embedding glove --embedding_dict "/mnt/hd3/ivo_data/embeddings/wang2vec_structured_skipgram.emb.gz" \
 --train "/mnt/hd3/ivo_data/datasets/HAREM/dev_harem_selective_iob2.conll" --dev "/mnt/hd3/ivo_data/datasets/HAREM/dev_harem_selective_iob2.conll"\
  --test "/mnt/hd3/ivo_data/datasets/HAREM/dev_harem_selective_iob2.conll"
