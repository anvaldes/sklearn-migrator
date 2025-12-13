#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(pwd)/integration/_artifacts"
mkdir -p "$WORKDIR"

docker build -t sklearn-migrator-env-in  -f integration/env_in/Dockerfile .
docker build -t sklearn-migrator-env-out -f integration/env_out/Dockerfile .

# 1) Serialize en env_in
docker run --rm \
  -v "$WORKDIR:/artifacts" \
  sklearn-migrator-env-in \
  python integration/scripts/serialize_rf_clf.py \
    --out-model /artifacts/model.json \
    --out-pred /artifacts/pred_in.npy

# 2) Deserialize + predict en env_out
docker run --rm \
  -v "$WORKDIR:/artifacts" \
  sklearn-migrator-env-out \
  python integration/scripts/deserialize_rf_clf.py \
    --in-model /artifacts/model.json \
    --out-pred /artifacts/pred_out.npy

# 3) Compare (puede ser en host)
python integration/scripts/compare_preds.py \
  --pred-in "$WORKDIR/pred_in.npy" \
  --pred-out "$WORKDIR/pred_out.npy"

echo "✅ Integration test passed!"
