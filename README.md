# NeurIPS 2020 paper

Code for the paper
"[Measuring Systematic Generalization in Neural Proof Generation with Transformers](https://arxiv.org/abs/2009.14786)"
![Measuring Systematic Generalization in Neural Proof Generation with Transformers](img/screenshot.png)

### Installation

````
pip install torch

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 
````

### Download data

Manually from this link:
https://drive.google.com/file/d/1TUXEcyR5i3TCrYtlTIX65swQH56TSiJj/view?usp=sharing
and unzip into the `./data/` folder

or

````
cd ./data && ./setup.sh
````

##### The file structure of the data is the following:
````yaml
data/
|  backward/  # proof sentences are reversed, from answer to facts (~backward chaining)
|  |  test/
|  |  train/
|  |  |  {long|short}_proof_1.{2|4|6}_train_{amt|facts}_anon.txt.4000
|  |  valid/
|  |  |  {long|short}_proof_1.{2|4|6}_valid_{amt|facts}_anon.txt.4000
|  forward/  # proof sentences are in order, from facts to answer (~forward chaining)
|  |  test/
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
