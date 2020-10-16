#!/usr/bin/env bash


if [ $1 == "sp" ]
then
  prefix="../data/forward"
  m1="short-proof"
  m2="short-proof"
elif [ $1 == "spr" ]
then
  prefix="../data/backward"
  m1="short-proof-rev"
  m2="short-proof"
elif [ $1 == "lp" ]
then
  prefix="../data/forward"
  m1="long-proof"
  m2="long-proof"
elif [ $1 == "lpr" ]
then
  prefix="../data/backward"
  m1="long-proof-rev"
  m2="long-proof"
elif [ $1 == "np" ]
then
  prefix="../data/forward"
  m1="no-proof"
  m2="no-proof"
else
  echo "ERROR: invalid parameter. $1 must be 'sp' or 'spr' or 'lp' or 'lpr' or 'np'."
fi


if [ $2 == "facts" ] || [ $2 == "amt" ]
then
  cd src
  for i in 2 # 3 4 5 6 7 8 9 10
  do
    if [ $m1 == "no-proof" ] || [ $i == 2 ] || [ $i == 3 ] || [ $i == 4 ]
    then
      bs=64
      ml=256
    elif [ $i == 5 ] || [ $i == 6 ] || [ $i == 7 ]
    then
      bs=32
      ml=512
    else
      bs=32
      ml=1024
    fi
    python generate_proofs-answers.py \
        --load_path "../models/1_${m1}_${2}_2_4_6/gpt_tiny_anon" \
        --bpe_codes "${prefix}/codes.4000" --bpe_vocab_path "${prefix}/vocab.4000" \
        --gpt_vocab_path "../models/1_${m1}_${2}_2_4_6/vocab.pkl" --sample "topk" \
        --test_file "${prefix}/test/${i}/queries_1.${i}_test_${2}_ANON.txt" \
        --out_file "${prefix}/test/${i}/gpt_tiny_anon_${m2}_1.${i}_test_${2}_ANON.txt" \
        --max_batch_size ${bs} --max_length ${ml} --verbose "no"
  done
else
  echo "ERROR: invalid parameter. $2 must be 'facts' or 'amt'."
fi


