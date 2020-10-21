# Measuring Systematic Generalization in Neural Proof Generation with Transformers

This repository shows how to reproduce results from the paper
"[Measuring Systematic Generalization in Neural Proof Generation with Transformers](https://arxiv.org/abs/2009.14786)"
![Measuring Systematic Generalization in Neural Proof Generation with Transformers](img/screenshot.png)

### Installation

Tested on this environment:
- Python 3.6.8
- CUDA Version: 10.2

````
pip install -r requirements.txt
git clone https://github.com/NVIDIA/apex 
````
NOTE: `apex` must be cloned in the current directory or manually added to the `$PYTHONPATH`.

### Download data

Download manually from this link:
https://drive.google.com/file/d/1TUXEcyR5i3TCrYtlTIX65swQH56TSiJj/view?usp=sharing
and unzip into the `./data/` folder

-or-

Run the following commands:
````
cd ./data
chmod +x setup.sh
./setup.sh
````

**The file structure of the data is the following**:
````yaml
data/
|  relations_store.yaml  # copied from https://github.com/facebookresearch/clutrr/tree/master/clutrr/store
|  rules_store.yaml      # copied from https://github.com/facebookresearch/clutrr/tree/master/clutrr/store
|  backward/  # proof sentences are reversed, from answer to facts (~backward chaining)
|  |  test/
|  |  |  {2|3|4|5|6|7|8|9|10}/
|  |  |  |  {long|short}_proof_1.{2|3|4|5|6|7|8|9|10}_test_facts_ANON.txt  # target used to evaluate
|  |  |  |  queries_1.{2|3|4|5|6|7|8|9|10}_test_{amt|facts}_ANON.txt       # prefix used to generate
|  |  train/
|  |  |  {long|short}_proof_1.{2|4|6}_train_{amt|facts}_anon.txt.4000
|  |  valid/
|  |  |  {long|short}_proof_1.{2|4|6}_valid_{amt|facts}_anon.txt.4000
|  forward/  # proof sentences are in order, from facts to answer (~forward chaining)
|  |  test/
|  |  |  {2|3|4|5|6|7|8|9|10}/
|  |  |  |  {long|short|no}_proof_1.{2|3|4|5|6|7|8|9|10}_test_facts_ANON.txt  # target used to evaluate
|  |  |  |  queries_1.{2|3|4|5|6|7|8|9|10}_test_{amt|facts}_ANON.txt          # prefix used to generate
|  |  train/
|  |  |  {long|no|short}_proof_1.{2|4|6}_train_{amt|facts}_anon.txt.4000
|  |  valid/
|  |  |  {long|no|short}_proof_1.{2|4|6}_valid_{amt|facts}_anon.txt.4000
#               ^                   ^              ^
#          long, short, or no       ^              ^
#           proof strategies        ^              ^
#                                   ^              ^
#               lvl 2, 4, 6 family stories         ^
#                                                  ^
#          family graph expressed with the 'facts' or 'amt' template
````

### Run experiments

#### (1) Training
Tested with this hardware:
- gpu: 6 * 32 Gb Tesla V100
- cpu: 6 * 16 Gb

````
#
# FACTS
#

# --no proof sentences
python launch_job.py --config configs/gpt_tiny.json --dataset clutrr1_no_proof_facts_2+clutrr1_no_proof_facts_4+clutrr1_no_proof_facts_6 --experiment_name gpt_tiny_anon

# --forward proof sentences
python launch_job.py --config configs/gpt_tiny.json --dataset clutrr1_long_proof_facts_2+clutrr1_long_proof_facts_4+clutrr1_long_proof_facts_6 --experiment_name gpt_tiny_anon
python launch_job.py --config configs/gpt_tiny.json --dataset clutrr1_short_proof_facts_2+clutrr1_short_proof_facts_4+clutrr1_short_proof_facts_6 --experiment_name gpt_tiny_anon

# --reversed proof sentences
python launch_job.py --config configs/gpt_tiny.json --dataset clutrr1_long-proof-rev_facts_2+clutrr1_long-proof-rev_facts_4+clutrr1_long-proof-rev_facts_6 --experiment_name gpt_tiny_anon
python launch_job.py --config configs/gpt_tiny.json --dataset clutrr1_short-proof-rev_facts_2+clutrr1_short-proof-rev_facts_4+clutrr1_short-proof-rev_facts_6 --experiment_name gpt_tiny_anon

#
# AMT
#

# --no proof sentences
python launch_job.py --config configs/gpt_tiny.json --dataset clutrr1_no_proof_amt_2+clutrr1_no_proof_amt_4+clutrr1_no_proof_amt_6 --experiment_name gpt_tiny_anon

# --forward proof sentences
python launch_job.py --config configs/gpt_tiny.json --dataset clutrr1_long_proof_amt_2+clutrr1_long_proof_amt_4+clutrr1_long_proof_amt_6 --experiment_name gpt_tiny_anon
python launch_job.py --config configs/gpt_tiny.json --dataset clutrr1_short_proof_amt_2+clutrr1_short_proof_amt_4+clutrr1_short_proof_amt_6 --experiment_name gpt_tiny_anon

# --reversed proof sentences
python launch_job.py --config configs/gpt_tiny.json --dataset clutrr1_long-proof-rev_amt_2+clutrr1_long-proof-rev_amt_4+clutrr1_long-proof-rev_amt_6 --experiment_name gpt_tiny_anon
python launch_job.py --config configs/gpt_tiny.json --dataset clutrr1_short-proof-rev_amt_2+clutrr1_short-proof-rev_amt_4+clutrr1_short-proof-rev_amt_6 --experiment_name gpt_tiny_anon
````

After training, you should see the following new files:
````
logs/
|  log_[experiment_name]_[dataset].log
models/
|  1_{no|short|long}-proof{-rev|}_{facts|amt}_2_4_6/
|  |  vocab.pkl
|  |  [experiment_name]/
|  |  |  args.pkl
|  |  |  cur_model.pt
|  |  |  cur_optimizer.pt
|  |  |  meta.pkl
|  |  |  meta2.json
|  |  |  model.pt
|  |  |  optimizer.pt
|  |  |  train.log
````

#### (2) Generation

Tested with this hardware:
- gpu: 1 * 12 Gb
- cpu: 2 * 4 Gb

Make the scripts executable:
````
chmod +x generate_proofs-answers.sh
chmod +x generate_answers.sh
````

**NOTE**: if you want to choose which file to generate to, modify the `--out_file` argument in `generate_proofs-answers.sh` and `generate_answers.sh`.
Otherwise your predictions will be stored by default in
`data/{forward|backward}/test/{2|3|4|5|6|7|8|9|10}/gpt_tiny_anon_{no|short|long}-proof_1.{2|3|4|5|6|7|8|9|10}_test_{facts|amt}_ANON.txt`
when generating both the proofs and answers; and in
`data/{forward|backward}/test/{2|3|4|5|6|7|8|9|10}/gpt_tiny_anon_{no|short|long}-proof_1.{2|3|4|5|6|7|8|9|10}_test_{facts|amt}_ANON_ans-only.txt`
when generating only the answers.

Run one of the following command to generate predictions on all test levels from 2 to 10:
````
#
# FACTS
#

# --no proof sentences
./generate_proofs-answers.sh np facts  # given facts story + query generate 'none' + answer
./generate_answers.sh np facts         # given facts story + query + 'none' generate answer

# --forward proof sentences
./generate_proofs-answers.sh lp facts  # given facts story + query generate long-proof + answer
./generate_proofs-answers.sh sp facts  # given facst story + query generate short-proof + answer
./generate_answers.sh lp facts  # given facst story + query + long-proof generate answer
./generate_answers.sh sp facts  # given facts story + query + short-proof generate answer

# --reversed proof sentences
./generate_proofs-answers.sh lpr facts  # given facts story + query generate long-proof-rev + answer
./generate_proofs-answers.sh spr facts  # given facts story + query generate short-proof-rev + answer
./generate_answers.sh lpr facts  # given facts story + query + long-proof-rev generate answer
./generate_answers.sh spr facts  # given facts story + query + short-proof-rev generate answer

#
# AMT
#

# --no proof sentences
./generate_proofs-answers.sh np amt  # given amt story + query generate 'none' + answer
./generate_answers.sh np amt         # given amt story + query + 'none' generate answer

# --forward proof sentences
./generate_proofs-answers.sh lp amt  # given amt story + query generate long-proof + answer
./generate_proofs-answers.sh sp amt  # given amt story + query generate short-proof + answer
./generate_answers.sh lp amt  # given amt story + query + long-proof generate answer
./generate_answers.sh sp amt  # given amt story + query + short-proof generate answer

# --reversed proof sentences
./generate_proofs-answers.sh lpr amt  # given amt story + query generate long-proof-rev + answer
./generate_proofs-answers.sh spr amt  # given amt story + query generate short-proof-rev + answer
./generate_answers.sh lpr amt  # given amt story + query + long-proof-rev generate answer
./generate_answers.sh spr amt  # given amt story + query + short-proof-rev generate answer
````

#### (3) Evaluation
Tested with this hardware requirements:
- cpu: 1 * 4 Gb

````
cd src
python evaluate_generation.py --truth <PATH_TO_GROUND_TRUTH_FILE> \
                              --pred  <PATH_TO_PREDICTIONS_FILE> \
                              --proof_type <"normal" -or- "reversed">
````

**NOTE**:
The evaluation script will save the answer accuracy and proof validity into a .yaml file at the same location as your `--pred` file.
For instance, if you passed `--pred ../data/test.txt`, the script will create the following file: `../data/test.yaml` with the following structure:
````yaml
{
  correct: {
    idx: [...]   # list of line numbers for which the answer is correct.
    score: 0.\#\#  # proportion (0<#<1) of accurate answers.
  }
  correct_but_invalid: {
    idx: [...]   # list of line numbers for which the answer is correct but the proof is inconsistent.
    score: 0.\#\#  # proportion (0<#<1) of accurate answers with an inconsistent proof.
  }
  wrong: {
    idx: [...]   # list of line numbers for which the answer is incorrect.
    score: 0.\#\#  # proportion (0<#<1) of inaccurate answers.
  }
  wrong_but_valid: {
    idx: [...]   # list of line numbers for which the answer is incorrect but the proof is consistent.
    score: 0.\#\#  # proportion (0<#<1) of inaccurate answers with an consistent proof.
  }
}
````

You can also look at `evaluate_generation.sh` for an example following the same naming scheme as in the `generate_proofs-answers.sh` and `generate_answers.sh` scripts.
This will automatically evaluate the generated files on all test levels for both proof+answer predictions and answer-only predictions if you previously ran `generate_proofs-answers.sh` and `generate_answers.sh`.
The evaluation files will then be stored by default in
`data/{forward|backward}/test/{2|3|4|5|6|7|8|9|10}/gpt_tiny_anon_{no|short|long}-proof_1.{2|3|4|5|6|7|8|9|10}_test_{facts|amt}_ANON.yaml`
when evaluating both the proofs and answers; and in
`data/{forward|backward}/test/{2|3|4|5|6|7|8|9|10}/gpt_tiny_anon_{no|short|long}-proof_1.{2|3|4|5|6|7|8|9|10}_test_{facts|amt}_ANON_ans-only.yaml`
when evaluating only the answers.

To do this, start by making the script executable:
````
chmod +x evaluate_generation.sh
````

And run one of the following command to evaluate your predictions:
````
#
# FACTS
#

# --no proof sentences
./evaluate_generation.sh np facts

# --forward proof sentences
./evaluate_generation.sh lp facts
./evaluate_generation.sh sp facts

# --reversed proof sentences
./evaluate_generation.sh lpr facts
./evaluate_generation.sh spr facts

#
# AMT
#

# --no proof sentences
./evaluate_generation.sh np amt

# --forward proof sentences
./evaluate_generation.sh lp amt
./evaluate_generation.sh sp amt

# --reversed proof sentences
./evaluate_generation.sh lpr amt
./evaluate_generation.sh spr amt
````

### Acknowledgements

We greatly thank Sandeep Subramanian (https://github.com/MaximumEntropy) for allowing us to use and share some of his experimental code, from which this repository was constructed.

### Cite

To cite our paper, please use the following bibtex:
````
@incollection{gontier2020measuring,
  title = {Measuring Systematic Generalization in Neural Proof Generation with Transformers},
  author = {Gontier, Nicolas and Sinha, Koustuv and Reddy, Siva and Pal, Christopher},
  booktitle = {Advances in Neural Information Processing Systems 33},
  year = {2020},
  publisher = {Curran Associates, Inc.},
  url = {https://arxiv.org/pdf/2009.14786.pdf}
}
````
