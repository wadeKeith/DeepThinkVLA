#!/bin/bash
# This script downloads the LIBERO CoT dataset for DeepThinkVLA experiments.
# Author: Cheng Yin
# Date: 2025-09
# Copyright (c) Cheng Yin. All rights reserved.
# See LICENSE file in the project root for license information.

set -euo pipefail

LOCAL_DIR=${1:-data/datasets/yinchenghust/libero_cot}
HF_REPO=${2:-yinchenghust/libero_cot}

mkdir -p "${LOCAL_DIR}"

huggingface-cli download \
    --repo-type dataset \
    --resume-download "${HF_REPO}" \
    --local-dir "${LOCAL_DIR}"
