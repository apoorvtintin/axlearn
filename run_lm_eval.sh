model=${1}
model_path=${2}
max_concurrent_req=${3:-1}
port=${4:-8000} 
task_name=${5:-"gsm8k_cot"}
results_dir=${6}

# source ~/venv-lmeval/bin/activate

echo "Running LM Eval Client for model: ${model}"

export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE="http://localhost:${port}/v1"

echo "Starting lm_eval"
# MODEL_PATH="Llama-2-7b-hf"
python -m lm_eval \
       --model local-completions \
       --tasks  ${task_name}\
       --model_args model=$model_path,trust_remote_code=True,base_url=http://localhost:${port}/v1/completions,tokenized_requests=False,tokenizer_backend="huggingface",use_fast_tokenizer=False,add_bos_token=True,num_concurrent=${max_concurrent_req} \
       --log_samples \
       --output_path ${results_dir} \
       --limit 10\
       --verbosity DEBUG
