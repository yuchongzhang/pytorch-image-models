#!/bin/bash
# =============================================================================
# ImageNet-1K Base Mixed DeiT Sweep Launcher
# =============================================================================
# Multi-GPU distributed training sweep for base mixed DeiT-3 models on
# ImageNet-1K using DeiT-3 paper hyperparameters (400 epochs, 3-augment, BCE).
#
# Usage:
#   ./imagenet_sweep_base_mixed.sh [--dry-run]
#
# Models: vanilla DeiT-3 Base + mixed variants with different Laplacian head counts
# Seeds: 3 random seeds (0, 1, 2)
# GPUs: 4x H100 (80GB) per job, no gradient checkpointing
# =============================================================================

# ========================
# SWEEP CONFIGURATION
# ========================

# Models to train: vanilla DeiT-3 Base + mixed variants with varying Laplacian heads
MODELS=(
    "deit3_base_qk_patch16_192"      # vanilla DeiT-3 Base (12 standard heads)
    "deit3_base_1L_patch16_192"
    "deit3_base_3L_patch16_192"
    "deit3_base_6L_patch16_192"
    "deit3_base_9L_patch16_192"
    "deit3_base_11L_patch16_192"
    "deit3_base_12L_patch16_192"
)

# Random seeds
SEEDS=(0 1 2)

# Drop path rates for this sweep
DROP_PATHS=(0.2)

# ========================
# TRAINING HYPERPARAMETERS
# ========================
# DeiT-3-style settings for base models at 192x192

EPOCHS=400
BATCH_SIZE=256              # Per-GPU batch size (conservative for H100 80GB without grad checkpointing)
VAL_BATCH_SIZE=256
GRAD_ACCUM_STEPS=2          # Effective global batch = 256 * 4 GPUs * 2 = 2048 (matches DeiT-3 paper)
LR_BASE=1e-3                # DeiT-3 paper base LR
LR_BASE_SIZE=1024           # DeiT-3 paper reference batch size for LR scaling
LR_BASE_SCALE="linear"     # CRITICAL: DeiT-3 uses linear scaling (timm defaults to sqrt for adamw)
WEIGHT_DECAY=0.05           # DeiT-3 paper
OPT="adamw"
SCHED="cosine"
WARMUP_EPOCHS=5
COOLDOWN_EPOCHS=0
MIN_LR=1e-6
IMG_SIZE=192                # DeiT-3 paper: ViT-L pre-trained at 192x192
CLIP_GRAD=1.0

# Augmentation (DeiT-3 paper: 3-augment + BCE)
AA="3a"                     # 3-augment (Solarize, Desaturate, GaussianBlur)
MIXUP=0.8
CUTMIX=1.0
SMOOTHING=0.1
COLOR_JITTER=0.3            # DeiT-3 paper with 3-augment
BCE_LOSS=true               # Binary Cross Entropy loss

# Efficiency settings
AMP_DTYPE="bfloat16"
COMPILE=true
COMPILE_MODE="default"
CHANNELS_LAST=true

# Model EMA (DeiT-3 paper)
MODEL_EMA=false
MODEL_EMA_DECAY=0.99996

# Checkpointing
CHECKPOINT_HIST=1           # Keep only the best checkpoint plus last.pth.tar

# ========================
# SLURM CONFIGURATION
# ========================
ACCOUNT="def-papyan"
PARTITION=""                # Leave empty for default
EXCLUDE_NODES="rg21802"     # Latest failures all landed here with only 3 visible GPUs
TIME="03:00:00"             # 3h wall time (auto-requeue handles longer training)
CPUS=32                     # CPUs for data loading
NUM_GPUS=4                  # Number of H100 GPUs per node
GPU_TYPE="h100:${NUM_GPUS}"
MEM="400G"
MIN_FREE_GPU_MEM_MB=76000   # Requeue if any allocated GPU has less free memory than this

# Paths
VENV_PATH="/home/yucz/links/projects/def-papyan/yucz/venvs/timm_env/bin/activate"
PROJECT_DIR="/home/yucz/links/projects/def-papyan/yucz/pytorch-image-models"
LOG_DIR="${PROJECT_DIR}/logs/imagenet_sweep_base_mixed"
OUTPUT_BASE="/home/yucz/links/scratch/timm_imagenet_sweep_base_mixed"
DATA_DIR="/home/yucz/links/projects/def-papyan/shared/imagenet-1k/data"   # Path to ImageNet parquet files
DATASET="parquet/"          # Use custom parquet reader for HF-format data

# Derived training settings
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS))
EFFECTIVE_LR=$(python3 -c "
lr_base = float('${LR_BASE}')
lr_base_size = int('${LR_BASE_SIZE}')
effective_batch = int('${EFFECTIVE_BATCH_SIZE}')
scale = '${LR_BASE_SCALE}'
if scale == 'linear':
    effective_lr = lr_base * effective_batch / lr_base_size
elif scale == 'sqrt':
    effective_lr = lr_base * (effective_batch / lr_base_size) ** 0.5
else:
    effective_lr = lr_base
print(f'{effective_lr:g}')
")

# ========================
# SCRIPT LOGIC
# ========================

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No jobs will be submitted ==="
fi

mkdir -p "${LOG_DIR}"

TOTAL_JOBS=$((${#MODELS[@]} * ${#SEEDS[@]} * ${#DROP_PATHS[@]}))
echo "Preparing to launch ${TOTAL_JOBS} ImageNet training jobs..."
echo "  Models: ${MODELS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Drop paths: ${DROP_PATHS[*]}"
echo "  GPUs per job: ${NUM_GPUS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Grad accum steps: ${GRAD_ACCUM_STEPS}"
echo "  Effective global batch: ${EFFECTIVE_BATCH_SIZE}"
echo "  Effective learning rate: ${EFFECTIVE_LR}"
echo "  Epochs: ${EPOCHS}"
echo "  Image size: ${IMG_SIZE}"
echo ""

JOB_COUNT=0
SKIPPED_FINISHED=0
SKIPPED_QUEUED=0
SUBMITTED_JOBS=()

# Returns 0 (true) if a job with this name is already queued or running in SLURM
is_queued() {
    squeue --me --name="${1}" --noheader 2>/dev/null | grep -q .
}

# Returns 0 (true) if the experiment has already completed the requested epochs
is_finished() {
    local exp_dir="${OUTPUT_BASE}/${1}"
    local summary_file="${exp_dir}/summary.csv"
    [[ -f "${summary_file}" ]] || return 1
    local completed
    completed=$(python3 -c "
import csv
try:
    max_epoch = -1
    with open('${summary_file}', newline='') as f:
        for row in csv.DictReader(f):
            epoch = row.get('epoch')
            if not epoch:
                continue
            max_epoch = max(max_epoch, int(float(epoch)))
    print(max_epoch + 1)
except Exception:
    print(0)
" 2>/dev/null)
    [[ "${completed}" -ge "${2}" ]]
}

for SEED in "${SEEDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for DROP_PATH in "${DROP_PATHS[@]}"; do
            JOB_COUNT=$((JOB_COUNT + 1))

            EXP_NAME="${MODEL}_seed${SEED}_dp${DROP_PATH}_lr=${EFFECTIVE_LR}_bs=${EFFECTIVE_BATCH_SIZE}"
            JOB_NAME="in1k_${MODEL}_s${SEED}_dp${DROP_PATH}"

            # Workers per GPU
            WORKERS=$((CPUS / NUM_GPUS))

            # Build training command with torchrun for distributed training
            TRAIN_CMD="torchrun --nproc_per_node=${NUM_GPUS} --master_port=\$(shuf -i 29500-29999 -n 1) train.py \\
    --model ${MODEL} \\
    --seed ${SEED} \\
    --img-size ${IMG_SIZE} \\
    --epochs ${EPOCHS} \\
    --batch-size ${BATCH_SIZE} \\
    --validation-batch-size ${VAL_BATCH_SIZE} \\
    --grad-accum-steps ${GRAD_ACCUM_STEPS} \\
    --opt ${OPT} \\
    --lr-base ${LR_BASE} \\
    --lr-base-size ${LR_BASE_SIZE} \\
    --lr-base-scale ${LR_BASE_SCALE} \\
    --weight-decay ${WEIGHT_DECAY} \\
    --sched ${SCHED} \\
    --warmup-epochs ${WARMUP_EPOCHS} \\
    --cooldown-epochs ${COOLDOWN_EPOCHS} \\
    --min-lr ${MIN_LR} \\
    --clip-grad ${CLIP_GRAD} \\
    --drop-path ${DROP_PATH} \\
    --dataset ${DATASET} \\
    --data-dir ${DATA_DIR} \\
    --train-split train \\
    --val-split validation \\
    --output ${OUTPUT_BASE} \\
    --experiment ${EXP_NAME} \\
    --workers ${WORKERS} \\
    --pin-mem \\
    --amp \\
    --amp-dtype ${AMP_DTYPE} \\
    --checkpoint-hist ${CHECKPOINT_HIST}"

            # Add compilation flags
            if [[ "${COMPILE}" == "true" ]]; then
                TRAIN_CMD="${TRAIN_CMD} \\
    --torchcompile inductor \\
    --torchcompile-mode ${COMPILE_MODE}"
            fi
            if [[ "${CHANNELS_LAST}" == "true" ]]; then
                TRAIN_CMD="${TRAIN_CMD} \\
    --channels-last"
            fi

            # Add Model EMA
            if [[ "${MODEL_EMA}" == "true" ]]; then
                TRAIN_CMD="${TRAIN_CMD} \\
    --model-ema \\
    --model-ema-decay ${MODEL_EMA_DECAY}"
            fi

            # Add augmentation args
            if [[ -n "${AA}" ]]; then
                TRAIN_CMD="${TRAIN_CMD} \\
    --aa ${AA}"
            fi
            if [[ -n "${MIXUP}" && "${MIXUP}" != "0.0" && "${MIXUP}" != "0" ]]; then
                TRAIN_CMD="${TRAIN_CMD} \\
    --mixup ${MIXUP}"
            fi
            if [[ -n "${CUTMIX}" && "${CUTMIX}" != "0.0" && "${CUTMIX}" != "0" ]]; then
                TRAIN_CMD="${TRAIN_CMD} \\
    --cutmix ${CUTMIX}"
            fi
            if [[ -n "${SMOOTHING}" && "${SMOOTHING}" != "0.0" && "${SMOOTHING}" != "0" ]]; then
                TRAIN_CMD="${TRAIN_CMD} \\
    --smoothing ${SMOOTHING}"
            fi
            if [[ -n "${COLOR_JITTER}" ]]; then
                TRAIN_CMD="${TRAIN_CMD} \\
    --color-jitter ${COLOR_JITTER}"
            fi

            # Add BCE loss
            if [[ "${BCE_LOSS}" == "true" ]]; then
                TRAIN_CMD="${TRAIN_CMD} \\
    --bce-loss"
            fi

            # Add wandb logging (offline mode — sync later with `wandb sync <run-dir>`)
            TRAIN_CMD="${TRAIN_CMD} \\
    --log-wandb \\
    --wandb-project imagenet-mixed-deit-sweep"

SLURM_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=${ACCOUNT}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gpus-per-node=${GPU_TYPE}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}.err
${EXCLUDE_NODES:+#SBATCH --exclude=${EXCLUDE_NODES}}
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
${PARTITION:+#SBATCH --partition=${PARTITION}}

# ---- Signal handling for auto-requeue ----
REQUEUED=0
handle_timeout() {
    echo "[\$(date)] Caught SIGUSR1 — requeueing job \$SLURM_JOB_ID"
    REQUEUED=1
    scontrol requeue "\$SLURM_JOB_ID"
    kill -TERM "\$TRAIN_PID" 2>/dev/null
    wait "\$TRAIN_PID" 2>/dev/null
}
trap 'handle_timeout' USR1

# ---- Load environment ----
module load gcc arrow/21.0.0
source ${VENV_PATH}

# Set environment variables for performance
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline

# Change to project directory
cd ${PROJECT_DIR}

# Sanity-check the allocated GPUs before launching torchrun.
VISIBLE_GPUS=\$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "[\$(date)] Host=\$(hostname -s) CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=\${SLURM_JOB_GPUS:-unset} SLURM_GPUS_ON_NODE=\${SLURM_GPUS_ON_NODE:-unset} torch.cuda.device_count()=\${VISIBLE_GPUS}"
if [[ "\${VISIBLE_GPUS}" -ne "${NUM_GPUS}" ]]; then
    echo "[\$(date)] Expected ${NUM_GPUS} visible GPUs but found \${VISIBLE_GPUS}; requeueing without launching torchrun."
    scontrol requeue "\$SLURM_JOB_ID"
    exit 0
fi

FREE_MEM_LINES=\$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)
echo "[\$(date)] GPU free memory (MiB):"
echo "\${FREE_MEM_LINES}"
LOW_MEM_GPUS=\$(printf '%s\n' "\${FREE_MEM_LINES}" | awk -F', *' -v min_mem="${MIN_FREE_GPU_MEM_MB}" '\$2 < min_mem {print \$1 ":" \$2}')
if [[ -n "\${LOW_MEM_GPUS}" ]]; then
    echo "[\$(date)] Found GPU(s) below ${MIN_FREE_GPU_MEM_MB} MiB free memory: \${LOW_MEM_GPUS}; requeueing without launching torchrun."
    scontrol requeue "\$SLURM_JOB_ID"
    exit 0
fi

# ---- Auto-resume from last checkpoint ----
RESUME_ARG=""
WANDB_RESUME_ARG=""
LAST_CKPT="${OUTPUT_BASE}/${EXP_NAME}/last.pth.tar"

# Generate a deterministic wandb run ID from experiment name (stable across requeues)
WANDB_RUN_ID=\$(python3 -c "import hashlib; print(hashlib.md5(b'${EXP_NAME}').hexdigest()[:8])")

if [[ -f "\${LAST_CKPT}" ]]; then
    echo "[\$(date)] Resuming from checkpoint: \${LAST_CKPT}"
    RESUME_ARG="--resume \${LAST_CKPT}"
    WANDB_RESUME_ARG="--wandb-resume-id \${WANDB_RUN_ID}"
fi

# ---- Run training ----
${TRAIN_CMD} \${RESUME_ARG} \${WANDB_RESUME_ARG} &
TRAIN_PID=\$!
wait \$TRAIN_PID
TRAIN_EXIT=\$?

if [[ \${REQUEUED} -eq 0 ]]; then
    exit \${TRAIN_EXIT}
fi
EOF
)

            echo "[${JOB_COUNT}/${TOTAL_JOBS}] ${EXP_NAME}"

            if is_finished "${EXP_NAME}" "${EPOCHS}"; then
                echo "  Skipping: already completed ${EPOCHS} epochs"
                SKIPPED_FINISHED=$((SKIPPED_FINISHED + 1))
                continue
            fi

            if is_queued "${JOB_NAME}"; then
                echo "  Skipping: job already queued/running in SLURM"
                SKIPPED_QUEUED=$((SKIPPED_QUEUED + 1))
                continue
            fi

            if [[ "${DRY_RUN}" == "true" ]]; then
                echo "  Would submit: ${JOB_NAME}"
                echo "  Command preview:"
                echo "    ${TRAIN_CMD}" | head -5
                echo "    ..."
            else
                JOB_ID=$(echo "${SLURM_SCRIPT}" | sbatch --parsable)
                SUBMITTED_JOBS+=("${JOB_ID}")
                echo "  Submitted: Job ID ${JOB_ID}"
            fi
        done
    done
done

echo ""
echo "=== Summary ==="
echo "Total jobs: ${JOB_COUNT}"
echo "Skipped (already finished): ${SKIPPED_FINISHED}"
echo "Skipped (already queued/running): ${SKIPPED_QUEUED}"
echo "Epochs per job: ${EPOCHS}"
echo "GPUs per job: ${NUM_GPUS}"
echo "Per-GPU batch: ${BATCH_SIZE}, grad accum: ${GRAD_ACCUM_STEPS}, effective global: ${EFFECTIVE_BATCH_SIZE}, effective LR: ${EFFECTIVE_LR}"
echo "Wall time per submission: ${TIME} (auto-requeue on timeout)"

if [[ "${DRY_RUN}" == "false" && ${#SUBMITTED_JOBS[@]} -gt 0 ]]; then
    echo "Submitted job IDs: ${SUBMITTED_JOBS[*]}"
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "Cancel all with: scancel ${SUBMITTED_JOBS[*]}"
fi
