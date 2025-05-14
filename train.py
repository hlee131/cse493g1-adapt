import argparse
import os
import shutil
import torch

from datasets import load_dataset
from trainset_generation import regenerate_data

from src.Trainer import LLMAgentTrainable

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a llama model.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--trained_model", type=str)
    parser.add_argument("--data_path", type=str, default="logs")
    parser.add_argument("--max_training_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, help="This overrides the max_training_steps", default=None)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.01)
    parser.add_argument("--trainer_type", type=str, default='dpo', choices=["sft", "dpo"])
    parser.add_argument("--only_generate_data", action='store_true') ## used to generate data as preprocessing before training
    parser.add_argument("--regenerate_data_at_start", action='store_true')
    parser.add_argument("--no_questions", action='store_true')
    parser.add_argument("--new_data_only", action='store_true')

    args = parser.parse_args()

    time_per_loop = 100000000
    
    if args.epochs is not None:
        args.max_training_steps = None
        
    if args.trained_model is None:
        args.trained_model = os.path.join(args.base_model+"_trained")
    
    if os.path.exists(args.trained_model) and args.base_model != args.trained_model:
        if 'model.pth' in os.listdir(args.trained_model):
            overwrite = input(f"Model already exists at {args.trained_model}. Press y to overwrite it?") == 'y'
            if not overwrite:
                print("Exiting...")
                exit()
        else:
            print(f"Model already exists, but looks incomplete, with only:\n{os.listdir(args.trained_model)} \nOverwriting...")
        shutil.rmtree(args.trained_model)
            
    
    model_trainable = None
    
    if args.base_model != args.trained_model:
        model_trainable = LLMAgentTrainable(overwrite_base_model=False, exact_outpath=args.trained_model, **args.__dict__)
        args.base_model = args.trained_model
    else:
        print("Loading trainable model...")
        model_trainable = LLMAgentTrainable(overwrite_base_model=True, **args.__dict__)
    print("Model loaded...")
    
    if args.regenerate_data_at_start:
        process_new = True
        regenerate_data(args.data_path, no_questions=args.no_questions)
        if args.only_generate_data:
            raise RuntimeError
    if args.trainer_type == 'dpo':
        data_filename = {split:os.path.join(args.data_path,f"gen_dataset_DPO_unpriv_verbose_{split}_onlyAction.jsonl") for split in ['train', 'test']}
        print(f"Using data from: \n  - {data_filename['train']}\n  - {data_filename['test']}")
        datasets = {split:load_dataset("json", data_files=data_filename[split])["train"] for split in ['train', 'test']}
        model_trainable.train_dpo(datasets)
    elif args.trainer_type == 'sft':
        dataset_name = 'SFT_correction'
        data_filename = {split:os.path.join(args.data_path,f"gen_dataset_{dataset_name}_unpriv_verbose_{split}_onlyAction.jsonl") for split in ['train', 'test']}
        print(f"Using data from: \n  - {data_filename['train']}\n  - {data_filename['test']}")
        datasets = {split:load_dataset("json", data_files=data_filename[split])["train"] for split in ['train', 'test']}
        model_trainable.train_sft(datasets)
    del model_trainable
    torch.cuda.empty_cache()
