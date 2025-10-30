export DATA_DIR="/projects/bodymaps/jliu452/Data/Dataset907_UCSF/PT_data"
export CHECKPOINT="./models/best_model_99.pth"   

python -m src.inference \
  --data_dir $DATA_DIR \
  --num_classes 4 \
  --checkpoint_path $CHECKPOINT \
  --multi_gpu \