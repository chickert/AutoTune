#!/bin/bash

# --- Configuration ---
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
benchmark_dir="$(cd "$script_dir/.." && pwd)"
script_path="$script_dir/get_detections.py"
day_start_time="2022-11-25 14:00:00"          # Start of the day's processing
day_end_time="2022-11-25 14:10:00"            # End of the day's processing
increment_minutes=10                          # Time increment in minutes
log_dir="$benchmark_dir/detector_measurements/logs"  # Directory for log files

timestamp_to_seconds() {
  if date -d "$1" +%s >/dev/null 2>&1; then
    date -d "$1" +%s
  else
    date -j -f "%Y-%m-%d %H:%M:%S" "$1" "+%s"
  fi
}

seconds_to_timestamp() {
  if date -d "@$1" "+%Y-%m-%d %H:%M:%S" >/dev/null 2>&1; then
    date -d "@$1" "+%Y-%m-%d %H:%M:%S"
  else
    date -j -f "%s" "$1" "+%Y-%m-%d %H:%M:%S"
  fi
}

# --- Main Processing ---
start_seconds=$(timestamp_to_seconds "$day_start_time")
end_seconds=$(timestamp_to_seconds "$day_end_time")
increment_seconds=$((increment_minutes * 60))

current_start_seconds=$start_seconds

# Create the log directory if it doesn't exist
mkdir -p "$log_dir"

while [[ "$current_start_seconds" -lt "$end_seconds" ]]; do
  current_start_ts=$(seconds_to_timestamp "$current_start_seconds")
  current_end_seconds=$((current_start_seconds + increment_seconds))
  current_end_ts=$(seconds_to_timestamp "$current_end_seconds")

  # Create log file name with start and end timestamps
  start_sanitized=$(echo "$current_start_ts" | sed 's/ /_/g; s/:/-/g')
  end_sanitized=$(echo "$current_end_ts" | sed 's/ /_/g; s/:/-/g')
  log_file="$log_dir/${start_sanitized}_${end_sanitized}.log"

  echo ""
  echo "Calling $script_path --start_time '$current_start_ts' --end_time '$current_end_ts' --disable_tqdm"
  echo "Logging to $log_file"

  # Run the Python script in the background (&) w/ tqdm disabled to avoid cluttering log file
  python "$script_path" --start_time "$current_start_ts" --end_time "$current_end_ts" --disable_tqdm > "$log_file" 2>&1 &

  echo "Launched python script in background for '$current_start_ts' to '$current_end_ts' with PID $!"

  current_start_seconds=$current_end_seconds
  sleep 2
done

echo "Runs launched."
