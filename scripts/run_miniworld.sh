# Set your API keys via environment variables before running:
# export WANDB_API_KEY=your_wandb_api_key
# export HF_TOKEN=your_huggingface_token (optional, for HuggingFace login)
# export AWS_ACCESS_KEY_ID=your_aws_key (optional, for S3 uploads)
# export AWS_SECRET_ACCESS_KEY=your_aws_secret (optional, for S3 uploads)

pip install "gymnasium[other]" qwen-vl-utils deepspeed==0.15.4 boto3==1.35.95
pip install transformers==4.50.3

# Optional: Login to HuggingFace if HF_TOKEN is set
if [ -n "$HF_TOKEN" ]; then
    python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
fi

cd /workspace/RL4VLM/VLM_PPO_miniworld

KL_BETA=${KL_BETA:-0.05}
PROMPT_VERSION=${PROMPT_VERSION:-"v3"}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1}
VALUE_LOSS_COEF=${VALUE_LOSS_COEF:-0.25}
INIT_LR=${INIT_LR:-5e-5}
END_LR=${END_LR:-5e-7}
GAE_LAMBDA=${GAE_LAMBDA:-0.99}
GAMMA=${GAMMA:-0.995}
TEMPERATURE=${TEMPERATURE:-0.2}
PPO_EPOCH=${PPO_EPOCH:-2}
MAX_IMAGE_OBS_LEN=${MAX_IMAGE_OBS_LEN:-4}
WANDB_PROJECT=${WANDB_PROJECT:-"mlc-miniworld-vsl-token-level-rebuttal-v2"}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-128}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2-VL-7B-Instruct"}
SEED=${SEED:-1}
VERSION=${VERSION:-"v1"}
ENV_NAME=${ENV_NAME:-"MiniWorld-Hallway-v0"}
RUN_NAME=$MODEL_PATH-$SEED-$ENV_NAME-$PROMPT_VERSION-$GAE_LAMBDA-$GAMMA-$TEMPERATURE-$PPO_EPOCH-$MAX_IMAGE_OBS_LEN-$MAX_EPISODE_STEPS

pip3 freeze 
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="0" xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" accelerate launch --config_file scripts/config_zero2.yaml --main_process_port 29380 main.py --modular --config configs/miniworld_qwen2vl.yaml --use-wandb --seed 42

    # --wandb-run v1-no_sfm_surprise-adaptive-lambda \
    # --save-interval 5 \
    # --save-path /workspace/checkpoints/miniworld/miniworld-hallway/v1-no_sfm_surprise-adaptive-lambda \
    # --log-dir /workspace/logs/v1-no_sfm_surprise-adaptive-lambda



# 0,1,2,3,4,5,6,7
#   --gae-lambda 0.99 \
#   --gamma 0.995 \