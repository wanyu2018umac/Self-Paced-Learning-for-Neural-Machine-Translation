python -u translate.py --prefix "" \
        --src_test ../data/WMT14_EnDe/newstest2014.bpe.en \
        --tgt_test ../data/WMT14_EnDe/newstest2014.bpe.de \
        --src_vocab $1src_vocab.pt --tgt_vocab $1tgt_vocab.pt --joint_vocab $1joint_vocab.pt \
        --model_prefix $1 --model_suffix ".pt" \
        --model $2 \
        --params $1params.txt \
        --batch_size 32 --infer_max_seq_length 50 --infer_max_seq_length_mode relative \
        --beam_size $3 --decoding_alpha $4 --device 0
