python inference.py --do_predict \
    --checkpoints ./result_config/checkpoint-200000 \
    --data_path ../benchmark/test \
    --output_dir ./test_output \
    --device cuda:0 \
    --mode artificial \