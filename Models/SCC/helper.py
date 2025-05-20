
def split_by_id_ratio_1_to_4(df):
    # 1. Count how many times each ID appears
    id_counts = df['ID'].value_counts().to_dict()

    # 2. Sort IDs by descending row count
    sorted_ids = sorted(id_counts.items(), key=lambda x: x[1], reverse=True)

    # 3. Assign every 5th ID to validation, rest to train
    train_ids, val_ids = set(), set()
    for i, (id_val, _) in enumerate(sorted_ids):
        if i % 5 == 0:
            val_ids.add(id_val)
        else:
            train_ids.add(id_val)

    # 4. Create train and validation sets
    df_train = df[df['ID'].isin(train_ids)].reset_index(drop=True)
    df_val = df[df['ID'].isin(val_ids)].reset_index(drop=True)

    # 5. Print class distribution
    for name, subset in [("Train", df_train), ("Validation", df_val)]:
        total = len(subset)
        class_counts = subset['label'].value_counts().to_dict()
        print(f"{name} set: {total} samples — class distribution: {class_counts}")

    return df_train, df_val