export NEEL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --num_processes 4 --num_machines=1 --mixed_precision='bf16' ft.py --output_dir /uac/gds/hqcao23/hqcao/gx/Protein_MSA_Fold/openfold32-large \
    --dataset_name openfold \
    --remove_unused_columns False \
    --do_train True \
    --overwrite_output_dir True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 200 \
    --max_steps 200000 \
    --learning_rate 6e-5 \
    --lr_scheduler_type polynomial \
    --warmup_ratio 0.01 \
    --weight_decay 1e-5 \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end True \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 20 \
    --prediction_loss_only True \
    --num_alignments 32 \
    --threshold 512 \
    --fp16 False \
    --gradient_accumulation_steps 4 \
    --bf16 True \
    --optim adamw_torch_fused \
    --max_grad_norm 1.0 \
    # --resume_from_checkpoint /uac/gds/hqcao23/hqcao/gx/Protein_MSA_Fold/openfold32-large/checkpoint-10000 \
    # --safe_serialization False \
    # --save_safetensors False \
    # --resume_from_checkpoint /uac/gds/hqcao23/hqcao/gx/Protein_MSA_Fold/openfold32/checkpoint-180000 \
    # --dynamo_backend='cudagraphs'
    # --use_cache False \
    # --no_cuda True # for debug
    # --train_file ['/uac/gds/hqcao23/hqcao/openfold/esm_msa/','/uac/gds/hqcao23/hqcao/openfold/uniclust_emb/'] \

