python -u main.py --file_prefix ../data/IWSLT15_EnVi/ \
  --src_train train.en --tgt_train train.vi \
  --src_valid dev.en --tgt_valid dev.vi \
  --save_vocab --bpe_src --bpe_tgt --num_of_steps 50000 \
  --update_decay 1 --device 0 \
  --batch_capacity 8192 \
  --share_embedding --share_projection_and_embedding \
  --layer_norm_pre learnable --layer_norm_post none \
  --layer_norm_encoder_start none --layer_norm_encoder_end learnable \
  --layer_norm_decoder_start none --layer_norm_decoder_end learnable \
  --embedding_keep_prob 0.7 --residual_keep_prob 0.7 --attention_keep_prob 0.7 --feedforward_keep_prob 0.7 \
  --src_pad_token "<PAD>" --src_unk_token "<UNK>" --src_sos_token "" --src_eos_token "<EOS>" \
  --tgt_pad_token "<PAD>" --tgt_unk_token "<UNK>" --tgt_sos_token "<PAD>" --tgt_eos_token "<EOS>" \
  --annotate "IWSLT15_EnVi" --num_of_workers 8 --max_save_models -1 \
  --eval_every_steps 100000 --save_every_steps 1000 --max_save_models -1 \
  --normalized_cl_factors --exponential_value 2
