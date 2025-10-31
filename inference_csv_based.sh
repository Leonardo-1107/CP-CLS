export DATA_DIR="/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro"
export CHECKPOINT="./models/best_model_99_smile.pth"   
export CSV="./input_folder/UCSF_88K_Test_Set_v2_sampled3000.csv"

python -m src.inference \
  --data_dir $DATA_DIR \
  --num_classes 4 \
  --slice_num 5 \
  --batch_size 2 \
  --checkpoint_path $CHECKPOINT \
  --csv_reference_path $CSV \
  --multi_gpu \