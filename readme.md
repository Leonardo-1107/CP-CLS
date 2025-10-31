<center>
<h1>SMILE CT Contrast Phase Classifier</h1>
</center>

This script performs CT phase classification using a pre-trained DenseNet121 model (from MONAI). It supports both **testing with labeled data** and **inference on unlabeled CT scans or CSV-listed datasets**.

2026 CVPR SMILE backbone.

---


**Hardware-light design**:
    
    (1) GPU minimum requirement: RTX 3050+, VRAM > 8GB.
    (2) Light local storage and CPU occupation.
    (3) model.pth size: ~ 3 MB.


<details>
<summary>📦 Environment & Dependencies (click to expand)</summary>

```bash
conda create -n kcls 
conda activate kcls
pip install torch monai scikit-learn matplotlib pandas joblib tqdm h5py numpy
```
</details>

## Inference Guidence
Two data formats are acceptable for this classifier, Direct and AbdomenAtlasPro form.

(A) **Directly inference** on the CT volume data:
```bash
input_ct_folder
    --{patient_id}.nii.gz
    ...
    --inference.csv <-inference result csv
```

the output format will be:
```
    Patient ID, Phase Label
    example_001, non-contrast
```

**Run the code with**
`bash inference.sh`

(B) Inference with **given CSV guidence**, and in **<mark>AbdomenAtlasPro</mark>** formula:
```bash
input_ct_folder
    --patient_001
        --ct.nii.gz
    --patient_002
        --ct.nii.gz

input_folder
    --csv_guidence.csv
    --csv_guidence_phase_pred.csv <-inference result csv
```

**Run the code with**
`bash inference_csv_based.sh`

### Parameter Setting

<details>
<summary>⚙️ Parameters (click to expand)</summary>

| **Argument** | **Type / Default** | **Description** |
|---------------|--------------------|-----------------|
| `--data_dir` | *str* | Path to the folder containing CT scans (`.nii.gz` files or patient subfolders). |
| `--checkpoint_path` | *str* | Path to the trained DenseNet checkpoint used for inference. |
| `--num_classes` | *int*, default=`4` | Number of phase classes (e.g., non-contrast, arterial, venous, delayed). |
| `--batch_size` | *int*, default=`8` | Number of slices processed per batch during inference. |
| `--slice_num` | *int*, default=`7` | Number of axial slices sampled from each CT volume. |
| `--csv_reference_path` | *str*, optional | CSV file containing patient IDs (`bdmap_id`) for indirect inference mode. |
| `--test_mode` | *flag* | Enables **test mode** (requires ground truth labels, outputs metrics and confusion matrix). |
| `--multi_gpu` | *flag* | Enables **multi-GPU inference** using `torch.nn.DataParallel()`. |

</details>

---

<details>
<summary>📖 Notes</summary>

Haha