#!/usr/bin/env bash
# Arrhenius GPU template. Override resource options with sbatch flags if needed.
#SBATCH --account=naiss2026-4-902-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --job-name=pp4
#SBATCH --output=pp4-%j.out
#SBATCH --error=pp4-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the pp4 project directory}"

# CONDA_BASE may be exported explicitly before sbatch. No installation in this job.
if [[ -z "${CONDA_BASE:-}" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$("$CONDA_EXE" info --base)"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    else
        echo 'Set CONDA_BASE to the existing conda installation directory before sbatch.' >&2
        exit 1
    fi
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate avalanche

# Respect Slurm device allocation. Do not pass host GPU numbers manually.
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose a GPU allocation}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u run_experiments.py "$@"
