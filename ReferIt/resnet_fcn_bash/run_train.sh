(./train.sh | tee stdout.log) 3>&1 1>&2 2>&3 | tee stderr.log
