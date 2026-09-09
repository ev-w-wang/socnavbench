#!/bin/bash
# GPU MPPI full 33-episode GT then ORCA suites at the 4096-sample profile.
set -u
ROOT="/home/everett/socnavbench/SocNavBench"
BASE="${ROOT}/tests/mppi_tuning/gpu4096"
GT_DIR="${ROOT}/tests/mppi_tuning/gpu4096_gt"
ORCA_DIR="${ROOT}/tests/mppi_tuning/gpu4096_orca"
POLICY="${ROOT}/policy.config"
STATUS="${BASE}/STATUS"

mkdir -p "${BASE}" "${GT_DIR}" "${ORCA_DIR}"
cp "${POLICY}" "${BASE}/policy.config"
cp "${ROOT}/params/episode_params_val.ini" "${BASE}/episode_params_val.ini"

# shellcheck source=/dev/null
source /home/everett/miniconda3/etc/profile.d/conda.sh
conda activate socnavbench
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/joystick"
export PYOPENGL_PLATFORM=egl

set_predictor() {
  python - "$POLICY" "$1" <<'PY'
import sys
path, method = sys.argv[1], sys.argv[2]
lines = []
replaced = False
for line in open(path):
    if line.startswith("dynamic_prediction_method"):
        lines.append("dynamic_prediction_method = {}\n".format(method))
        replaced = True
    else:
        lines.append(line)
if not replaced:
    raise SystemExit("missing dynamic_prediction_method")
open(path, "w").writelines(lines)
PY
}

if [ -d "${ROOT}/tests/socnav/test_MPPI" ]; then
  rm -rf "${ROOT}/tests/socnav/test_MPPI_archive_before_gpu4096"
  mv "${ROOT}/tests/socnav/test_MPPI" "${ROOT}/tests/socnav/test_MPPI_archive_before_gpu4096"
fi

run_suite() {
  local label="$1"
  local logdir="$2"
  mkdir -p "${logdir}"
  rm -f /tmp/socnavbench_joystick_recv /tmp/socnavbench_joystick_send
  echo "START_${label} $(date -Is)" | tee -a "${STATUS}"
  python tests/test_episodes.py >/dev/null 2>&1 &
  local sim_pid=$!
  echo "${sim_pid}" >"${logdir}/simulator.pid"
  sleep 8
  python joystick/joystick_client.py --algo mppi >/dev/null 2>&1 &
  local joy_pid=$!
  echo "${joy_pid}" >"${logdir}/joystick.pid"
  wait "${sim_pid}"
  local sim_ec=$?
  wait "${joy_pid}"
  local joy_ec=$?
  {
    echo "DONE_${label} $(date -Is)"
    echo "${label}_simulator_exit=${sim_ec}"
    echo "${label}_joystick_exit=${joy_ec}"
  } | tee -a "${STATUS}"
  python tests/summarize_mppi_results.py \
    --root tests/socnav/test_MPPI \
    --output "${logdir}/mppi_summary" \
    >"${logdir}/mppi_summary.txt" 2>&1
  rm -rf "${logdir}/test_MPPI"
  if [ -d "${ROOT}/tests/socnav/test_MPPI" ]; then
    mv "${ROOT}/tests/socnav/test_MPPI" "${logdir}/test_MPPI"
  fi
  cp "${POLICY}" "${logdir}/policy.config"
  return "${sim_ec}"
}

echo "START $(date -Is)" | tee "${STATUS}"
python -c 'import torch; print("torch", torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")' | tee -a "${STATUS}"
python -c 'from joystick_py.joystick_mppi import load_mppi_config; c=load_mppi_config("policy.config"); print("samples={num_samples} horizon={horizon} iters={num_iterations} ang_frac={initial_sequence_angular_fraction}".format(**c))' | tee -a "${STATUS}"

set_predictor ground_truth
run_suite gt "${GT_DIR}"
GT_EC=$?

set_predictor orca
run_suite orca "${ORCA_DIR}"
ORCA_EC=$?

if [ -d "${ROOT}/tests/socnav/test_MPPI_archive_before_gpu4096" ]; then
  rm -rf "${ROOT}/tests/socnav/test_MPPI"
  mv "${ROOT}/tests/socnav/test_MPPI_archive_before_gpu4096" "${ROOT}/tests/socnav/test_MPPI"
fi

echo "GT_EC=${GT_EC} ORCA_EC=${ORCA_EC}" | tee -a "${STATUS}"
echo "SUMMARY $(date -Is)" | tee -a "${STATUS}"
exit "${GT_EC}"
