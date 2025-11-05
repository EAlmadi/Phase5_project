# tf_data_pipeline.py
import os
import random
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

# CONFIG
ROOT = "BreaKHis_v1/histology_slides/breast"
MAG = "40X"      # change to "100X","200X","400X" or None to use all
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

def gather_filepaths(root, mag=None):
    items = []  # list of tuples (filepath, label, patient_id)
    for label_name, label in [("benign", 0), ("malignant", 1)]:
        class_root = os.path.join(root, label_name)
        # patient folders are under class_root/SOB/...
        # Walk two levels: find patient directories under class_root
        for dirpath, dirnames, filenames in os.walk(class_root):
            # Identify patient folder names: they look like SOB_...
            # We'll treat any folder that contains a magnification subfolder as patient
            parts = dirpath.split(os.sep)
            # if the current directory equals a patient folder, it will have subfolder '40X' etc.
            if mag:
                mag_dir = os.path.join(dirpath, mag)
                if os.path.isdir(mag_dir):
                    for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff"):
                        for f in glob(os.path.join(mag_dir, ext)):
                            patient = os.path.basename(dirpath)  # patient id
                            items.append((f, label, patient))
            else:
                # include all magnifications: look for files directly under any magnification child
                for child in os.listdir(dirpath):
                    if child.endswith("X") and os.path.isdir(os.path.join(dirpath, child)):
                        mag_dir = os.path.join(dirpath, child)
                        for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff"):
                            for f in glob(os.path.join(mag_dir, ext)):
                                patient = os.path.basename(dirpath)
                                items.append((f, label, patient))
    return items

def split_by_patient(items, val_frac=0.15, test_frac=0.15, seed=SEED):
    """
    Split items (path, label, patient) into train/val/test sets.
    Ensures that patients are grouped together and that the split
    is stratified by the majority label of each patient.
    """
    # Group by patient
    patient_data = {}
    for path, label, patient in items:
        if patient not in patient_data:
            patient_data[patient] = []
        patient_data[patient].append((path, label))

    # Determine each patient's dominant label (benign/malignant)
    patient_labels = []
    for patient, pairs in patient_data.items():
        labels = [l for _, l in pairs]
        dominant_label = max(set(labels), key=labels.count)  # majority label per patient
        patient_labels.append((patient, dominant_label))

    # Split patients (not images!) with stratification
    patients = [p for p, _ in patient_labels]
    labels = [l for _, l in patient_labels]

    # First: train+val vs test
    trainval_pat, test_pat = train_test_split(
        patients, test_size=test_frac, stratify=labels, random_state=seed
    )

    # Now: split train vs val
    train_pat, val_pat = train_test_split(
        trainval_pat, test_size=val_frac / (1 - test_frac),
        stratify=[labels[patients.index(p)] for p in trainval_pat],
        random_state=seed
    )

    # Collect files
    def collect(patient_set):
        out = []
        for p in patient_set:
            out.extend(patient_data[p])
        return out

    train = collect(train_pat)
    val = collect(val_pat)
    test = collect(test_pat)

    print(f"Patients -> Train: {len(train_pat)}, Val: {len(val_pat)}, Test: {len(test_pat)}")
    return train, val, test

def preprocess_image(path, label, img_size=IMG_SIZE):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    # simple augmentations for histology
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, label

def make_dataset(pairs, batch=BATCH, shuffle=True, augment_data=False):
    paths = [p for p,l in pairs]
    labels = [l for p,l in pairs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    if augment_data:
        ds = ds.map(lambda x,y: augment(x,y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds

if __name__ == "__main__":
    items = gather_filepaths(ROOT, mag=MAG)   # set mag=None to collect all magnifications
    print("Total images collected:", len(items))
    train_pairs, val_pairs, test_pairs = split_by_patient(items)
    print("Train/Val/Test counts:", len(train_pairs), len(val_pairs), len(test_pairs))
    train_ds = make_dataset(train_pairs, augment_data=True)
    val_ds = make_dataset(val_pairs, augment_data=False)
    test_ds = make_dataset(test_pairs, augment_data=False)
