python -u translate.py --prefix "" \
        --src_test ../data/WMT14_EnDe/newstest2013.bpe.en \
        --tgt_test ../data/WMT14_EnDe/newstest2013.bpe.de \
        --src_vocab $1src_vocab.pt --tgt_vocab $1tgt_vocab.pt --joint_vocab $1joint_vocab.pt \
        --model_prefix $1 --model_suffix ".pt" \
        --model $(seq -s " " $2 $3 $4) \
        --params $1params.txt \
        --batch_size 32 --infer_max_seq_length 50 --infer_max_seq_length_mode relative \
        --beam_size $5 --decoding_alpha $6 --device 0
