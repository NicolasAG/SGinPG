#!/usr/bin/env bash


###############################

# ---------------- TRAIN / VALID --------------------
for fold in "train" "valid"
do
  echo "for 1.2345678910_${fold}.csv"
  echo "  no_proof:"
  python csv2txt.py \
      --csv_in /data/1.2345678910_${fold}/1.2345678910_${fold}.csv \
      --out_prefix /data/1.2345678910_${fold}/no_proof_1.2345678910_${fold} \
      --query_only 'false' \
      --proof_type 'none' \
      --gender_aware 'no' \
      --reversed 'yes'

  echo "  long_proof:"
  python csv2txt.py \
      --csv_in /data/1.2345678910_${fold}/1.2345678910_${fold}.csv \
      --out_prefix /data/1.2345678910_${fold}/long_proof_1.2345678910_${fold} \
      --query_only 'false' \
      --proof_type 'long' \
      --gender_aware 'no' \
      --reversed 'yes'

  echo "  short_proof:"
  python csv2txt.py \
      --csv_in /data/1.2345678910_${fold}/1.2345678910_${fold}.csv \
      --out_prefix /data/1.2345678910_${fold}/short_proof_1.2345678910_${fold} \
      --query_only 'false' \
      --proof_type 'short' \
      --gender_aware 'no' \
      --reversed 'yes'

  echo "  queries only:"
  python csv2txt.py \
      --csv_in /data/1.2345678910_${fold}/1.2345678910_${fold}.csv \
      --out_prefix /data/1.2345678910_${fold}/queries_1.2345678910_${fold} \
      --query_only 'true' \
      --gender_aware 'no' \
      --reversed 'yes'

done
# ------------------------------------------

# ---------------- TEST --------------------

for i in 2 3 4 5 6 7 8 9 10
do
  echo "for 1.${i}_test.csv"
  echo "  no_proof:"
  python csv2txt.py \
      --csv_in '/data/1.${i}_test/1.${i}_test.csv' \
      --out_prefix '/data/1.${i}_test/no_proof_1.${i}_test' \
      --query_only 'false' \
      --proof_type 'none' \
      --gender_aware 'no' \
      --reversed 'yes'

  echo "  long_proof:"
  python csv2txt.py \
      --csv_in '/data/1.${i}_test/1.${i}_test.csv' \
      --out_prefix '/data/1.${i}_test/long_proof_1.${i}_test' \
      --query_only 'false' \
      --proof_type 'long' \
      --gender_aware 'no' \
      --reversed 'yes'

  echo "  short_proof:"
  python csv2txt.py \
      --csv_in '/data/1.${i}_test/1.${i}_test.csv' \
      --out_prefix '/data/1.${i}_test/short_proof_1.${i}_test' \
      --query_only 'false' \
      --proof_type 'short' \
      --gender_aware 'no' \
      --reversed 'yes'

  echo "  queries only:"
  python csv2txt.py \
      --csv_in '/data/1.${i}_test/1.${i}_test.csv' \
      --out_prefix '/data/1.${i}_test/queries_1.${i}_test' \
      --query_only 'yes' \
      --gender_aware 'no' \
      --reversed 'yes'

done

