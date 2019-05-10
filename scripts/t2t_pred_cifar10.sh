DATA_DIR=t2t_data/cifar10
OUTPUT_DIR=t2t_train/cifar10
PROBLEM=image_cifar10_plain_gen_rev
MODEL=imagetransformer
HPARAMS_SET=imagetransformer_cifar10_base

tensor2tensor/bin/t2t-decoder \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --problem=$PROBLEM \
    --hparams_set=$HPARAMS_SET \
    --model=$MODEL \
    --hparams="sampling_method=random" \
    --decode_hparams="beam_size=1,num_samples=12,batch_size=4,extra_length=3071"
