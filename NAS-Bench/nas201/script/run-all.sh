#!/bin/bash

tmux new-session -d -s "run1" 'sh run.sh 2 G1e119f168f06267d2312a61caf3f8b18'

echo "running run1 task"

sleep 5

tmux new-session -d -s "run2" 'sh run.sh 3 G4045a3b82a785e2578cef43679f9ed47'

echo "running run2 task"
sleep 5
