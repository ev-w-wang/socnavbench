#!/bin/bash
# Run the full GPU MPPI suite with trajectory-aligned uncertainty diagnostics.
set -u

ROOT="/home/everett/socnavbench/SocNavBench"
OUT="${ROOT}/tests/mppi_tuning/ellipse_full33"
POLICY="${ROOT}/policy.config"
ACTIVE_OUTPUT="${ROOT}/tests/socnav/test_MPPI"
WORK="$(mktemp -d)"
SIM_PID=""
JOY_PID=""

mkdir -p "${OUT}"
cp "${POLICY}" "${WORK}/policy.config"
if [ -d "${ACTIVE_OUTPUT}" ]; then
  mv "${ACTIVE_OUTPUT}" "${WORK}/preexisting_test_MPPI"
fi

cleanup() {
  if [ -n "${SIM_PID}" ]; then kill "${SIM_PID}" 2>/dev/null || true; fi
  if [ -n "${JOY_PID}" ]; then kill "${JOY_PID}" 2>/dev/null || true; fi
  rm -rf "${ACTIVE_OUTPUT}"
  if [ -d "${WORK}/preexisting_test_MPPI" ]; then
    mv "${WORK}/preexisting_test_MPPI" "${ACTIVE_OUTPUT}"
  fi
  cp "${WORK}/policy.config" "${POLICY}"
  rm -f /tmp/socnavbench_joystick_recv /tmp/socnavbench_joystick_send
  rm -rf "${WORK}"
}
trap cleanup EXIT INT TERM

source /home/everett/miniconda3/etc/profile.d/conda.sh
conda activate socnavbench
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/joystick"
export PYOPENGL_PLATFORM=egl

# Pin the tested ellipsoid profile while leaving temporal planning unchanged.
python - "${POLICY}" <<'PY'
import re
import sys

path = sys.argv[1]
values = {
    "dynamic_prediction_method": "orca",
    "dynamic_uncertainty_growth": "0.10",
    "dynamic_uncertainty_longitudinal_scale": "1.0",
    "dynamic_uncertainty_lateral_scale": "0.5",
    "dynamic_uncertainty_sigma_level": "2.0",
}
with open(path) as source:
    text = source.read()
for key, value in values.items():
    text, count = re.subn(
        r"(?m)^{}(\s*)=.*$".format(re.escape(key)),
        "{} = {}".format(key, value),
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("missing policy key: {}".format(key))
with open(path, "w") as output:
    output.write(text)
PY
cp "${POLICY}" "${OUT}/policy.config"

rm -rf "${ACTIVE_OUTPUT}" "${OUT}/diagnostics"
rm -f /tmp/socnavbench_joystick_recv /tmp/socnavbench_joystick_send
echo "START_NAVIGATION $(date -Is)" | tee "${OUT}/STATUS"

python tests/test_episodes.py >"${OUT}/simulator.log" 2>&1 &
SIM_PID=$!
sleep 8
python joystick/joystick_client.py --algo mppi >"${OUT}/joystick.log" 2>&1 &
JOY_PID=$!

wait "${SIM_PID}"
SIM_EC=$?
SIM_PID=""
wait "${JOY_PID}"
JOY_EC=$?
JOY_PID=""
echo "DONE_NAVIGATION $(date -Is) simulator=${SIM_EC} joystick=${JOY_EC}" \
  | tee -a "${OUT}/STATUS"
if [ "${SIM_EC}" -ne 0 ] || [ "${JOY_EC}" -ne 0 ]; then
  exit 1
fi

python tests/summarize_mppi_results.py \
  --root "${ACTIVE_OUTPUT}" \
  --output "${OUT}/mppi_summary" \
  >"${OUT}/mppi_summary.txt" 2>&1
SUMMARY_EC=$?
if [ "${SUMMARY_EC}" -ne 0 ]; then
  exit 1
fi

echo "START_DIAGNOSTICS $(date -Is)" | tee -a "${OUT}/STATUS"
python tests/mppi_tuning/evaluate_orca_uncertainty.py \
  --navigation-summary "${OUT}/mppi_summary.json" \
  --observation-interval 0.5 \
  --policy-config "${OUT}/policy.config" \
  --output-dir "${OUT}/diagnostics" \
  >"${OUT}/diagnostics.log" 2>&1
DIAGNOSTICS_EC=$?
echo "DONE_DIAGNOSTICS $(date -Is) diagnostics=${DIAGNOSTICS_EC}" \
  | tee -a "${OUT}/STATUS"
if [ "${DIAGNOSTICS_EC}" -ne 0 ]; then
  exit 1
fi

rm -f "${OUT}/simulator.log" "${OUT}/joystick.log"
echo "ALL_DONE $(date -Is)" | tee -a "${OUT}/STATUS"
