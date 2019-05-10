DATA_DIR=t2t_data/cifar10
OUTPUT_DIR=t2t_train/cifar10
PROBLEM=image_cifar10_plain_gen_rev
MODEL=imagetransformer
HPARAMS_SET=imagetransformer_cifar10_base

tensor2tensor/bin/t2t-trainer \
    --generate_data \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --problem=$PROBLEM \
    --hparams_set=$HPARAMS_SET \
    --model=$MODEL \
    --train_steps=100000 \
    --eval_steps=100
