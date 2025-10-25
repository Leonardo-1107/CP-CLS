# SMILE Phase Classification Inference

This script performs **CT phase classification** using a pre-trained DenseNet121 model (from MONAI).  
It supports both **testing with labeled data** and **inference on unlabeled CT scans or CSV-listed datasets**.

---

## 🧩 Environment

**Requirements:**
- Python 3.8+
- PyTorch
- MONAI
- scikit-learn
- matplotlib
- pandas
- joblib
- tqdm
- h5py
- numpy

Install dependencies (example):
```bash
pip install torch monai scikit-learn matplotlib pandas joblib tqdm h5py numpy
```


## Inference
Expect Data Format

(A) Directly inference on the CT volume data:
```
input_ct_folder
    --{patient_id}.nii.gz
    ...
    --inference.csv <-inference result csv
```

(B) Inference with given CSV guidence, and in AbdomenAtlasPro formula:
```
input_ct_folder
    --patient_001
        --ct.nii.gz
    --patient_002
        --ct.nii.gz

input_folder
    --csv_guidence.csv
    --csv_guidence_phase_pred.csv <-inference result csv
```