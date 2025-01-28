#!/bin/bash

# File to store the output
OUTPUT_FILE="gpu_cpu.txt"

# Clear the file if it exists
> "$OUTPUT_FILE"

# Run the program 20 times and save outputs
for i in {1..20}
do
    echo "Run $i:" >> "$OUTPUT_FILE"
    ./gpu_cpu >> "$OUTPUT_FILE" 2>&1
    echo -e "\n" >> "$OUTPUT_FILE"
done

echo "Finished running the ./gpu_cpu 20 times. Outputs saved in $OUTPUT_FILE."
