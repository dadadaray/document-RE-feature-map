#! /bin/bash
<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES=1
=======
export CUDA_VISIBLE_DEVICES=0
>>>>>>> dd6e4e7 (First commit)

# -------------------Training Shell Script--------------------
if true; then
  transformer_type=bert
  channel_type=context-based
  if [[ $transformer_type == bert ]]; then
    bs=2
    bl=3e-5
    uls=(4e-4)
<<<<<<< HEAD
    accum=1
=======
    accum=2
>>>>>>> dd6e4e7 (First commit)
    for ul in ${uls[@]}
    do
    python -u ./train_balanceloss.py --data_dir ./dataset/Re-DocRED\
    --channel_type $channel_type \
    --bert_lr $bl \
    --transformer_type $transformer_type \
    --model_name_or_path bert-base-cased \
    --train_file train_annotated.json \
    --dev_file dev.json \
    --test_file test.json \
    --train_batch_size $bs \
    --test_batch_size $bs \
    --gradient_accumulation_steps $accum \
    --num_labels 3 \
    --learning_rate $ul \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --adam_epsilon 1e-8\
    --num_train_epochs 30 \
    --seed 66 \
    --num_class 97 \
    --weight_decay 0\
    --save_path ./checkpoint/docred/train_bert-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.pt \
    --log_dir ./logs/docred/train_bert-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.log
    done
  elif [[ $transformer_type == roberta ]]; then
    type=context-based
    bs=2
<<<<<<< HEAD
    bls=(3e-5 2e-4)
    ul=4e-4
    accum=2
    for bl in ${bls[@]}
    do
    python -u ./train_balanceloss.py --data_dir ./dataset/docred \
    --channel_type $channel_type \
    --bert_lr $bl \
=======
    bls=6e-5
    uls=(4e-4)
    accum=2
    for ul in ${uls[@]}
    do
    python -u ./train_balanceloss_roberta.py --data_dir ./dataset/Re-DocRED \
    --channel_type $channel_type \
    --bert_lr $bls \
>>>>>>> dd6e4e7 (First commit)
    --transformer_type $transformer_type \
    --model_name_or_path ./roberta \
    --train_file train_annotated.json \
    --dev_file dev.json \
    --test_file test.json \
    --train_batch_size $bs \
    --test_batch_size $bs \
    --gradient_accumulation_steps $accum \
    --num_labels 4 \
    --learning_rate $ul \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 30 \
<<<<<<< HEAD
    --seed 111 \
    --num_class 97 \
    --save_path ./checkpoint/docred/train_roberta-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.pt \
    --log_dir ./logs/docred/train_roberta-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.log
=======
    --seed 66 \
    --num_class 97 \
    --save_path ./checkpoint/docred/train_roberta-lr${bls}_accum${accum}_unet-lr${ul}_type_${channel_type}.pt \
    --log_dir ./logs/docred/train_roberta-lr${bls}_accum${accum}_unet-lr${ul}_type_${channel_type}.log
>>>>>>> dd6e4e7 (First commit)
    done
  fi
fi



