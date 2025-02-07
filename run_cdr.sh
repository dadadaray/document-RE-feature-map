#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

<<<<<<< HEAD
if true; then
type=context-based
bs=2
bl=3e-5
uls=(4e-4)
accum=1
for ul in ${uls[@]}
do
python -u  ./train_bio.py --data_dir ./dataset/cdr \
  --max_height 35 \
  --channel_type $type \
  --bert_lr $bl \
  --transformer_type bert \
  --model_name_or_path allenai/scibert_scivocab_cased \
  --train_file train_filter.data \
  --dev_file dev_filter.data \
  --test_file test_filter.data \
  --train_batch_size $bs \
  --test_batch_size $bs \
  --gradient_accumulation_steps $accum \
  --num_labels 1 \
  --learning_rate $ul \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs 30 \
  --seed 111 \
  --num_class 2 \
  --save_path ./checkpoint/cdr/train_scibert-lr${bl}_accum${accum}_unet-lr${ul}_bs${bs}.pt \
  --log_dir ./logs/cdr/train_scibert-lr${bl}_accum${accum}_unet-lr${ul}_bs${bs}.log
done
fi

=======
# -------------------Training Shell Script--------------------
if true; then
  transformer_type=bert
  channel_type=context-based
  if [[ $transformer_type == bert ]]; then
    bs=2
    bl=3e-5
    uls=(3e-5)
    accum=2
    for ul in ${uls[@]}
    do
    python -u ./train_balanceloss_cdr.py --data_dir ./dataset/cdr_zhuanhuan/output\
    --channel_type $channel_type \
    --bert_lr $bl \
    --transformer_type $transformer_type \
    --model_name_or_path bert-base-cased \
    --train_file train_filter.json \
    --dev_file dev_filter.json \
    --test_file test_filter.json \
    --train_batch_size $bs \
    --test_batch_size $bs \
    --gradient_accumulation_steps $accum \
    --num_labels 2 \
    --learning_rate $ul \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --adam_epsilon 1e-8\
    --num_train_epochs 30 \
    --seed 128 \
    --num_class 2 \
    --weight_decay 0\
    --save_path ./checkpoint/cdr/train_bert-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.pt \
    --log_dir ./logs/cdr/train_bert-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.log
    done
  elif [[ $transformer_type == roberta ]]; then
    type=context-based
    bs=2
    bls=6e-5
    uls=(4e-4)
    accum=2
    for ul in ${uls[@]}
    do
    python -u ./train_balanceloss_roberta.py --data_dir ./dataset/Re-DocRED \
    --channel_type $channel_type \
    --bert_lr $bls \
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
    --seed 66 \
    --num_class 2 \
    --save_path ./checkpoint/docred/train_roberta-lr${bls}_accum${accum}_unet-lr${ul}_type_${channel_type}.pt \
    --log_dir ./logs/docred/train_roberta-lr${bls}_accum${accum}_unet-lr${ul}_type_${channel_type}.log
    done
  fi
fi



>>>>>>> dd6e4e7 (First commit)
