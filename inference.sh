export DATA_DIR="/projects/bodymaps/jliu452/Data/Dataset905_xinze/PT_data"
export CHECKPOINT="./models/best_model_89_smile.pth"   

python -m src.inference \
  --data_dir $DATA_DIR \
  --num_classes 4 \
  --checkpoint_path $CHECKPOINT