#!/bin/bash

for i in {1..100}  # Adjust the number of iterations as needed
do
    echo "Starting run $i"
    python3 ./Models/Transformer.py  # Replace with the actual script name
    sleep 10  # Optional: Pause for 10 seconds between runs
done
