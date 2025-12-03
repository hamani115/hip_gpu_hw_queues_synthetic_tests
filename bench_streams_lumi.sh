set -u

# === Compile ===
hipcc --offload-arch="gfx90a" stream_test.cpp -o stream_test_lumi_newdriver

# === Config ===
HIP_BIN=./stream_test_lumi_newdriver
OUT_CSV=results_hwq_streams_lumi_newdriver_rocm7-1-0_1000kl_10r.csv

# List of GPU_MAX_HW_QUEUES values to test
max_hw_queues_list=(1 2 4 8 16 24 32)

# List of num_streams values to test
num_streams_list=(1 2 4 8 16 24 32)

# How many repetitions per configuration
repetitions=4

# === Init CSV ===
echo "gpu_max_hw_queues,num_streams,run_id,time_ms" > "$OUT_CSV"

# === Sweep ===
for hwq in "${max_hw_queues_list[@]}"; do
    export GPU_MAX_HW_QUEUES="$hwq"
    echo "=== GPU_MAX_HW_QUEUES=$GPU_MAX_HW_QUEUES ==="

    for s in "${num_streams_list[@]}"; do
        for r in $(seq 1 "$repetitions"); do
            echo "Running: hwq=$hwq, streams=$s, rep=$r"

            # Run HIP program and capture stdout+stderr
            out="$("$HIP_BIN" "$s" 2>&1)"
            status=$?

            if [ $status -ne 0 ]; then
                echo "WARNING: $HIP_BIN exited with status $status for hwq=$hwq, streams=$s, rep=$r"
                echo "Output was:"
                echo "$out"
                # Skip this run but continue the sweep
                continue
            fi

            # Optional: uncomment this to see the program output each time
            # echo "$out"

            # Find the line containing the timing info
            # Matches e.g.
            #   "Total execution time: 0.418311 ms"
            #   "Total GPU time (H2D ...): 12.34 ms"
            time_line=$(printf '%s\n' "$out" | grep -E 'execution time|GPU time' | head -n1 || true)

            if [ -z "$time_line" ]; then
                echo "WARNING: Could not find timing line for hwq=$hwq, streams=$s, rep=$r"
                echo "Output was:"
                echo "$out"
                continue
            fi

            # Extract the numeric value before 'ms'
            time_ms=$(printf '%s\n' "$time_line" | grep -Eo '[0-9.]+[[:space:]]*ms' | head -n1 | awk '{print $1}')

            if [ -z "$time_ms" ]; then
                echo "WARNING: Failed to parse time from timing line:"
                echo "$time_line"
                continue
            fi

            # Append to CSV
            echo "${hwq},${s},${r},${time_ms}" >> "$OUT_CSV"
        done
    done
done

echo "Done. Results stored in: $OUT_CSV"

