export DATA_DIR="/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro"
export CHECKPOINT="./models/best_model_89_smile.pth"   
export CSV="./input_folder/UCSF-Test-Normal-Pancreas.csv"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m src.inference \
  --data_dir $DATA_DIR \
  --num_classes 4 \
  --slice_num 5 \
  --batch_size 2 \
  --checkpoint_path $CHECKPOINT \
  --csv_reference_path $CSV \
  --multi_gpu \