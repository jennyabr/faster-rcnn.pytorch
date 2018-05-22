run_file.sh:
python path_to_orig
python path_to_dev
# add line that shuts down the machine

running run_file.sh:
tmux
bash run_file.sh > log.txt
ctrl + D (hides tmux)
tmux attach -t (opens tmux)
