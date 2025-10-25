# Training Workflow

## Steps

1. **Edit experiment name** in `train.run`:
   ```bash
   # Example: "paper-2025-06"
   ```

2. **Submit training job**:
   ```bash
   sbatch train.run
   ```

3. **Monitor in MLflow**:
   - Experiment name: `paper-2025-06_training`
   - Each scenario logged with ID
   - Metrics, parameters, and artifacts tracked

## Experiment Naming Convention

- Manual experiment names: `paper-2025-06`
- Training experiment: `paper-2025-06_training`
- Inpainting experiment: `paper-2025-06_inpainting`

## Scenario Management

Scenarios defined in `epiframework.py` using dataclasses:
- UNet architecture configuration
- Dataset selection
- Transform specifications
- All parameters logged to MLflow
