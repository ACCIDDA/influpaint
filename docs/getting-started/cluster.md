# Run notebooks and jobs on UNC Longleaf

Complete [repository and environment setup](installation.md) first. Use your own UNC username and the path to your research checkout in the commands and forms below.

## Open OnDemand

Open [UNC Open OnDemand](https://ondemand.rc.unc.edu) and start a Jupyter session with a GPU request. Set the **Jupyter startup directory** to your research repository root, for example `/nas/longleaf/home/YOUR_ONYEN/influpaint`, and select the **Python (diffusion_torch)** kernel.

The early experiments used these additional job submission arguments:

```text
--mem=32gb -p volta-gpu --qos=gpu_access --gres=gpu:1
```

These are historical resource settings. The research notes report that a 16 GB Volta was insufficient for later runs and that Open OnDemand sessions could receive a small A100 MIG slice. Request enough GPU memory for the intended workload; the paper training launcher uses `l40-gpu`. Check the resources allocated to your session before starting training or inpainting.

## Jupyter on a compute node

For a terminal-managed session, request a GPU node, then start Jupyter from the research repository root in your environment:

```bash
srun --ntasks=1 --cpus-per-task=4 --mem=32G --time=18:00:00 \
  --partition=l40-gpu --gres=gpu:1 --qos=gpu_access --pty bash
conda activate diffusion_torch
cd /nas/longleaf/home/YOUR_ONYEN/influpaint
hostname
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Wait for Slurm to allocate the node. On your local machine, replace `COMPUTE_NODE` with the hostname printed above and create the SSH tunnel through the login node:

```bash
ssh -N -J YOUR_ONYEN@longleaf.unc.edu \
  -L 8888:127.0.0.1:8888 YOUR_ONYEN@COMPUTE_NODE
```

Open the local Jupyter URL printed by the server, including its token. To use a password, run `jupyter server password` once in the environment before starting the server. The earlier README referred to `create_notebook_password.sh` and `runjupyter.sh`; those launch scripts are not present under those names in this checkout.

The original UNC-IDD patron-node example used partition `jlessler` with one GPU for 18 hours. The recorded node specifications were 512 GB RAM, 56 physical CPU cores (112 with hyperthreading), and four NVIDIA L40 GPUs with 48 GB each. Use that partition only with access to the allocation.

## Paper batch jobs

The current launchers and command list are in `main_training/`. They contain the original author's Python environment path and Longleaf resource requests. Update those settings for your account and submit from the research repository root. Follow [training](../workflows/training.md), [forecast-job preparation](../workflows/forecast-jobs.md), and [inpainting](../workflows/inpainting.md) for the commands and required inputs.
