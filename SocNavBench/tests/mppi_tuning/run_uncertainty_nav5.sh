#!/bin/bash
# Compare zero, trajectory-aligned, and circular Gaussian uncertainty.
set -u

ROOT="/home/everett/socnavbench/SocNavBench"
OUT="${ROOT}/tests/mppi_tuning/uncertainty_nav5"
POLICY="${ROOT}/policy.config"
EPISODES="${ROOT}/params/episode_params_val.ini"
ACTIVE_OUTPUT="${ROOT}/tests/socnav/test_MPPI"
WORK="$(mktemp -d)"
SIM_PID=""
JOY_PID=""

mkdir -p "${OUT}"
cp "${POLICY}" "${WORK}/policy.config"
cp "${EPISODES}" "${WORK}/episode_params_val.ini"
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
  cp "${WORK}/episode_params_val.ini" "${EPISODES}"
  rm -f /tmp/socnavbench_joystick_recv /tmp/socnavbench_joystick_send
  rm -rf "${WORK}"
}
trap cleanup EXIT INT TERM

# Restrict the simulator to the same five scenes used for calibration.
python - "${EPISODES}" <<'PY'
import re
import sys

path = sys.argv[1]
scenarios = [
    "t_ETH1",
    "t_eth_dense_against",
    "t_zara1_dense_cross",
    "t_univ_trapBL",
    "t_dhotel_bottopmid",
]
with open(path) as source:
    text = source.read()
replacement = "tests=[\n{}\n    ]".format(
    "".join("    {!r},\n".format(name) for name in scenarios)
)
text, count = re.subn(
    r"(?ms)^tests=\[.*?^\s*\]",
    replacement,
    text,
    count=1,
)
if count != 1:
    raise SystemExit("unable to replace episode list")
with open(path, "w") as output:
    output.write(text)
PY

# shellcheck source=/dev/null
source /home/everett/miniconda3/etc/profile.d/conda.sh
conda activate socnavbench
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/joystick"
export PYOPENGL_PLATFORM=egl

set_profile() {
  python - "${POLICY}" "$1" "$2" "$3" <<'PY'
import re
import sys

path, growth, longitudinal, lateral = sys.argv[1:]
values = {
    "dynamic_prediction_method": "orca",
    "dynamic_uncertainty_growth": growth,
    "dynamic_uncertainty_longitudinal_scale": longitudinal,
    "dynamic_uncertainty_lateral_scale": lateral,
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
}

run_profile() {
  local label="$1"
  local growth="$2"
  local longitudinal="$3"
  local lateral="$4"
  local profile_dir="${OUT}/${label}"

  mkdir -p "${profile_dir}"
  rm -rf "${ACTIVE_OUTPUT}"
  rm -f /tmp/socnavbench_joystick_recv /tmp/socnavbench_joystick_send
  set_profile "${growth}" "${longitudinal}" "${lateral}"
  cp "${POLICY}" "${profile_dir}/policy.config"
  echo "START_PROFILE ${label} $(date -Is)" | tee -a "${OUT}/STATUS"

  python tests/test_episodes.py >"${profile_dir}/simulator.log" 2>&1 &
  SIM_PID=$!
  sleep 8
  python joystick/joystick_client.py --algo mppi >"${profile_dir}/joystick.log" 2>&1 &
  JOY_PID=$!

  wait "${SIM_PID}"
  local sim_ec=$?
  SIM_PID=""
  wait "${JOY_PID}"
  local joy_ec=$?
  JOY_PID=""

  python tests/summarize_mppi_results.py \
    --root "${ACTIVE_OUTPUT}" \
    --output "${profile_dir}/mppi_summary" \
    >"${profile_dir}/mppi_summary.txt" 2>&1
  local summary_ec=$?
  rm -rf "${ACTIVE_OUTPUT}"
  if [ "${sim_ec}" -eq 0 ] && [ "${joy_ec}" -eq 0 ] && [ "${summary_ec}" -eq 0 ]; then
    rm -f "${profile_dir}/simulator.log" "${profile_dir}/joystick.log"
  fi
  echo "DONE_PROFILE ${label} sim=${sim_ec} joystick=${joy_ec} summary=${summary_ec}" \
    | tee -a "${OUT}/STATUS"
  if [ "${sim_ec}" -ne 0 ] || [ "${joy_ec}" -ne 0 ] || [ "${summary_ec}" -ne 0 ]; then
    return 1
  fi
  return 0
}

rm -f "${OUT}/STATUS"
cp "${EPISODES}" "${OUT}/episode_params_val.five.ini"

run_profile zero 0.0 1.0 0.5 || exit 1
run_profile ellipse 0.10 1.0 0.5 || exit 1
run_profile circle 0.10 1.0 1.0 || exit 1

python tests/mppi_tuning/summarize_uncertainty_nav5.py \
  >"${OUT}/comparison.txt"
echo "ALL_PROFILES_DONE $(date -Is)" | tee -a "${OUT}/STATUS"
