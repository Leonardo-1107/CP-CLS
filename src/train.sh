DATA_DIR="/projects/bodymaps/jliu452/Data/Dataset904_SMILECLS/train"
SAVE_PATH="../logs"
CHECKPOINT="../models/best_model.pth"   


export OMP_NUM_THREADS=2
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --data_dir $DATA_DIR \
  --num_classes 4 \
  --save_path $SAVE_PATH \
  --resolution 512 \
  --num_epochs 100000 \
  --validation_interval_step 1000 \
  --batch_size 32 \
  --resume_from_checkpoint "/projects/bodymaps/jliu452/MONAI_CLS/models/best_model_89_smile.pth"