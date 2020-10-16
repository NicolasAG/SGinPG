#!/usr/bin/env bash

if [ $1 == "sp" ]
then
  prefix="../data/forward"
  m1="short-proof"
  m2="short_proof"
  proof_type="normal"
elif [ $1 == "spr" ]
then
  prefix="../data/backward"
  m1="short-proof-rev"
  m2="short_proof"
  proof_type="reversed"
elif [ $1 == "lp" ]
then
  prefix="../data/forward"
  m1="long-proof"
  m2="long_proof"
  proof_type="normal"
elif [ $1 == "lpr" ]
then
  prefix="../data/backward"
  m1="long-proof-rev"
  m2="long_proof"
  proof_type="reversed"
elif [ $1 == "np" ]
then
  prefix="../data/forward"
  m1="no-proof"
  m2="no_proof"
  proof_type="normal"
else
  echo "ERROR: invalid parameter. $1 must be 'sp' or 'spr' or 'lp' or 'lpr' or 'np'."
fi

if [ $2 == "facts" ] || [ $2 == "amt" ]
then
  cd src
  for i in 2 # 3 4 5 6 7 8 9 10
  do
    #  PROOF + ANSWER
    python evaluate_generation.py \
          --truth "${prefix}/test/${i}/${m2}_1.${i}_test_facts_ANON.txt" \
          --pred "${prefix}/test/${i}/gpt_tiny_anon_${m1}_1.${i}_test_${2}_ANON.txt" \
          --proof_type ${proof_type}
    # ANSWER ONLY
    python evaluate_generation.py \
          --truth "${prefix}/test/${i}/${m2}_1.${i}_test_facts_ANON.txt" \
          --pred "${prefix}/test/${i}/gpt_tiny_anon_${m1}_1.${i}_test_${2}_ANON_ans-only.txt" \
          --proof_type ${proof_type}
  done
else
  echo "ERROR: invalid parameter. $2 must be 'facts' or 'amt'."
fi
