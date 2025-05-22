export BASE_DIR=""
export HF_MODEL_NAME="unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

for SPLIT in 0 1 2 3; do 
	python run_dataset_gen.py \
		--run_config cfg/split${SPLIT}_run_config_train.json \
	        --out_dir data \
    	        --model_name_planner $HF_MODEL_NAME \
		--model_name_base    $HF_MODEL_NAME
done
