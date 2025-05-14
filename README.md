
This repository contains code for Actively Discovering and Adapting to Preferences for any Task (ADAPT) -- a benchmark designed to evaluate agents' ability to adhere to user preferences across various household tasks through active questioning, and Reflection-DPO, to finetune LLMs to the task of active questioning. Refer to our paper ["ADAPT: Actively Discovering and Adapting to Preferences for any Task"](https://arxiv.org/abs/2504.04040) for more details


# Setup and Usage

### Installation

Setup conda environment:

```
conda env create -f env.yml -n adapt
conda activate adapt
```

Install third-party requirements (with a fix to support locally downloaded Llama models):

```
git submodule update --init --recursive

cd third-party/transformers-CFG
git apply ../transformers-CFG.patch
cd ../..
pip install -e third-party/transformers-CFG
pip install protobuf==3.20.3
```


### Evaluation on ADAPT

To run evaluations for a model on all cross-validation splits for both seen and unseen persona:

Define `model_name_planner` as the name or path to model that you want to evaluate.

```
for split in 0 1 2 3
do
    for gen in seen_persona unseen_persona
    do
        python run_eval.py \
                --generalization_category $gen \
                --crossvalidation_split $split \
                --logs_dir logs \
                --model_name_planner $model_name_planner \
                --model_name_base meta-llama/Llama-3.1-70B-Instruct
    done
done
```

The following flags can be used to simualte baselines and ablations:
- `--interleave_thought`: To simulate a ReAct baseline, which interleaves thought with action prediction
- `--force_question_every_n 2`: To force the model to ask a question every 'N' steps. Setting to '2' will ask one question before each action
- `--no_ask_option`: Forbid the model from asking questions
- `--use_privileged_prior`: Simulate the privileged teacher by giving the model access to ground truth preferences


### Dataset Generation for Training

```
for split in 0 1 2 3
do
    python run_dataset_gen.py \
                    --run_config cfg/split$split"_run_config_train.json" \
                    --out_dir data \
                    --model_name_planner meta-llama/Llama-3.1-70B-Instruct \
                    --model_name_base meta-llama/Llama-3.1-70B-Instruct
done
```

### Model Training

Define `trained_model` as the path to the model being output, and `logs_dirname` as the directory containing the training data.

```
python train.py \
        --base_model meta-llama/Llama-3.1-70B-Instruct \
        --trained_model $trained_model \
        --data_path $logs_dirname
```


# Cite

If you find this work useful, consider citing us at:

```
@article{patel2025adapt,
  title={ADAPT: Actively Discovering and Adapting to Preferences for any Task},
  author={Patel, Maithili and Puig, Xavier and Desai, Ruta and Mottaghi, Roozbeh and Chernova, Sonia and Truong, Joanne and Rai, Akshara},
  journal={arXiv preprint arXiv:2504.04040},
  year={2025}
}
```
