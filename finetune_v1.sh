accelerate launch --num_processes 8 --num_machines=1 --mixed_precision='bf16' --dynamo_backend='cudagraphs' finetune.py --output_dir ./plame-v2-pdb \
    --dataset_name openfold \
    --train_file /uac/gds/hqcao23/hqcao/openfold/esm_msa/train \
    --remove_unused_columns False \
    --do_train True \
    --overwrite_output_dir True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 20 \
    --max_steps 200000 \
    --learning_rate 5e-5 \
    --lr_scheduler_type polynomial \
    --warmup_ratio 0.001 \
    --weight_decay 1e-5 \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end True \
    --evaluation_strategy steps \
    --eval_steps 2500 \
    --save_strategy steps \
    --save_steps 2500 \
    --save_total_limit 10 \
    --prediction_loss_only True \
    --num_alignments 32 \
    --threshold 512 \
    --fp16 False \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --max_grad_norm 1.0 \
    # --gradient_checkpointing True \
    # --use_cache False \
    # --no_cuda True # for debug
