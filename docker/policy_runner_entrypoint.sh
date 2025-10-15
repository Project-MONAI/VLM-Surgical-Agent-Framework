#!/bin/bash
set -euo pipefail

CONFIG_PATH=${POLICY_RUNNER_CONFIG:-/workspace/configs/policy_runner.yaml}

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Policy runner config not found at ${CONFIG_PATH}" >&2
    exit 1
fi

if grep -q "__REQUIRED__" "${CONFIG_PATH}"; then
    echo "Policy runner config ${CONFIG_PATH} still contains __REQUIRED__ placeholders. Please update it before running." >&2
    exit 1
fi

exec python3 -m policy_runner.run_policy --config "${CONFIG_PATH}" "$@"
