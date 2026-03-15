# Build one sample in dataset

OPENSWE_ROOT="/path/to/openswe"
DATA_PATH="/path/to/dataset"
OUTPUT_DIR="/path/to/logs"
SETUP_DIR="/path/to/repocache"
RESULT_DIR="/path/to/results"

DATA_PATH="/path/to/dataset.jsonl"
SELECTED_SAMPLE_ID="django--django--1024"

ROUND_LIMIT=4

export PYTHONPATH=$OPENSWE_ROOT:$PYTHONPATH
export OPENAI_KEY="sk-xxxxxxxxxxxxxxxxxxxxT3BlbkFJxxxxxxxxxxxxxxxxxxxx"
export OPENAI_API_BASE_URL="https://api.openai.com/v1/"
export CUSTOM_OPENAI_MODEL_NAME="gpt-5.4"
export DOCKER_REPOSITORY="registry.hub.docker.com/example/repo"

python -u $OPENSWE_ROOT/app/main.py swe-bench \
  --model custom-openai-model \
  --tasks-map $DATA_PATH \
  --task $SELECTED_SAMPLE_ID \
  --conv-round-limit $ROUND_LIMIT \
  --output-dir $OUTPUT_DIR \
  --setup-dir $SETUP_DIR \
  --results-path $RESULT_DIR
