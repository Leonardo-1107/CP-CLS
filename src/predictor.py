@torch.no_grad()
def predict_multi_patient(model, patient_ids, data_dir, slice_num, batch_size, device, chunk_size=100, n_jobs=8):
    """
    Chunk version to boost speed
    
    """
    patient_preds = {}

    # split into chunks of chunk_size
    for chunk_start in range(0, len(patient_ids), chunk_size):
        chunk_ids = patient_ids[chunk_start:chunk_start + chunk_size]
        all_datasets, patient_slices = [], {}

        def load_one(bid):
            ct_path = os.path.join(data_dir, bid, "ct.nii.gz")
            if not os.path.exists(ct_path):
                return None
            ds = CLSInferenceDataset(ct_path, slice_num=slice_num)
            return (bid, ds, len(ds))

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(load_one)(bid) for bid in tqdm(chunk_ids, desc="Loading datasets in parallel")
        )

        start = 0
        for r in results:
            if r is None:
                continue
            bid, ds, l = r
            all_datasets.append(ds)
            end = start + l
            patient_slices[bid] = (start, end)
            start = end

        if not all_datasets:
            continue

        concat_ds = ConcatDataset(all_datasets)
        dl = DataLoader(concat_ds, batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)

        preds = []
        for batch in tqdm(dl, desc=f"Inference chunk {chunk_start // chunk_size + 1}/{len(patient_ids) // chunk_size + 1}"):
            imgs = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast():
                out = model(imgs)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
        preds = np.array(preds)

        for bid, (s, e) in patient_slices.items():
            p = preds[s:e]
            if len(p) == 0:
                continue
            voted = int(np.bincount(p).argmax())
            patient_preds[bid] = voted

        torch.cuda.empty_cache()  # free memory after each chunk

    return patient_preds