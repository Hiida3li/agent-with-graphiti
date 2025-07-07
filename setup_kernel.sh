#!/bin/bash
ENV_NAME=${1:-default-env}
DISPLAY_NAME=${2:-"Python ($ENV_NAME)"}

python -m ipykernel install --user --name=$ENV_NAME --display-name "$DISPLAY_NAME"

bash setup_kernel.sh orki-env "Python (orki-env)"
