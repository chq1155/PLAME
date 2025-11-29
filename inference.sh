python inference_v0.py --do_predict \
    --checkpoints ./openfold32/checkpoint-200000 \
    --data_path /uac/gds/hqcao23/hqcao/openfold/enzyme \
    --output_dir /uac/gds/hqcao23/hqcao/gx/Protein_MSA_Fold/enzyme_plame \
    --device cuda:0 \
    --mode artificial \
    --num_alignments 100 \
    --augmentation_times 1 \
    --trials_times 1 \
    --repetition_penalty 1.0 \
    --temperature 1.0 \
    --top_p 0.95 \
    --device "cuda" \
    --do_sample True \
    --num_beams 1 \
    --num_beam_groups 1 \
    # --zero_shot True \
    # --plame False \
