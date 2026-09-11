# Mobile Deployment Benchmark

This directory contains the deployment evaluation of the Q5_K_M Edge AI model on the Samsung SM-G780F Android smartphone.

## Final sustained-load benchmark

The main benchmark consists of 102 inference runs:

- 6 fixed prompts
- 17 cycles
- 102 total executions
- 102/102 successful runs

Main results:

- Median TTFT: 632.644 ms
- P95 TTFT: 783.297 ms
- Median total inference latency: 1078.295 ms
- P95 total inference latency: 1271.071 ms
- Mean generation throughput: 35.397 tokens/s
- Peak resident memory: 1.379 GiB
- Battery temperature: 33.6 C to 38.9 C

The sustained-load analysis showed only limited performance degradation toward the end of the benchmark.

## Model initialization

Model loading was evaluated using one first load after device reboot and five subsequent application process restarts.

- Post-reboot load: 2931.458 ms
- Warm-load mean: 2065.220 ms
- Warm-load median: 2005.264 ms
- Warm-load reduction relative to post-reboot: 29.55%

The terms post-reboot and warm process restart are used because Android filesystem caches were not explicitly cleared.

## Offline evaluation

Offline operation was tested under Airplane mode.

- 30 requested inference runs
- 30 completed inference runs
- 0 failures
- airplane_mode_on = 1
- mDataConnectionState = 0

Wi-Fi was disabled during inference and re-enabled only after benchmark completion to retrieve logs through ADB.

## Main files

- `runs.csv`: per-inference measurements
- `summary.json`: overall benchmark statistics
- `thermal_battery.csv`: sustained-load thermal and battery samples
- `sustained_load_analysis.json`: early-versus-late performance analysis
- `model_load_benchmark.csv`: post-reboot and warm model-load measurements
- `offline_test_summary.txt`: offline test summary
- `device_info.json`: target-device information
- `initialization.json`: model and context initialization timing
- `raw/`: raw device and benchmark evidence

## Reproduction

`benchmark_main.dart` is a dedicated benchmark entry point and is separate from the normal application entry point (`main.dart`).

Example benchmark build command:

    flutter build apk --debug -t lib/benchmark_main.dart --dart-define=BENCHMARK_CYCLES=17

The normal application is unaffected and continues to use `lib/main.dart`.
