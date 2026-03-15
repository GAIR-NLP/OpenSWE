# Parallel Task Execution System

A master-worker system for distributed task processing with dynamic configuration and Prometheus monitoring.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Components

### 1. Master (`master.py`)
- **Role**: Orchestrates task distribution.
- **Function**: Scans a source directory for tasks and dispatches them to a shared queue directory.
- **Usage**: `python3 master.py master.yaml`

### 2. Worker (`worker.py`)
- **Role**: Executes tasks.
- **Function**: Monitors the shared queue, spawns multiple processes to run tasks, and handles logging/metrics.
- **Dynamic Config**: Supports live updates to process count and API settings via `dynamic_config.yaml`.
- **Usage**: `python3 worker.py -c work.yaml --taskdir <queue_dir> --datadir <output_dir>`

## Workflow

1. **Initialization**: Start the Master to begin dispatching tasks from the source data directory to the shared queue.
2. **Processing**: Start one or more Workers to consume tasks from the shared queue.
3. **Execution**: Each Worker spawns sub-processes to run the command defined in `work.yaml`.
4. **Monitoring**: Both components export metrics to Prometheus (ports configured in YAML).
5. **Dynamic Scaling**: Modify `dynamic_config.yaml` in the worker's data directory to adjust the number of concurrent processes without restarting.

## Configuration

- `master.yaml`: Master settings (directories, node count, buffer size).
- `work.yaml`: Initial worker settings (command, ports, intervals).
- `dynamic_config.yaml`: Runtime settings (API keys, weights, process count).
