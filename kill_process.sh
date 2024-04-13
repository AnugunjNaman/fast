for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i 4); do
    kill -9 $pid
done
