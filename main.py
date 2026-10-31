import pathlib
import sys

import cv2
import insightface.app
import numpy as np
import onnxruntime
import sklearn.cluster


def cluster_faces(
    npz_path: str | pathlib.Path = "face_embeddings.npz",
    eps: float = 0.4,
    min_samples: int = 2,
):
    """Loads the compressed NumPy binary face database, performs DBSCAN clustering

    on the 512-dim embeddings, and groups matching faces into person clusters.
    """
    file_path = pathlib.Path(npz_path)
    if not file_path.exists():
        print(f"Error: Database file '{npz_path}' not found. Run the scanner first.")
        return

    print(f"Loading embeddings from {npz_path}...")
    data = np.load(file_path, allow_pickle=True)

    image_paths: np.ndarray = data["paths"]
    bboxes: np.ndarray = data["bboxes"]
    embeddings: np.ndarray = data["embeddings"]

    if len(embeddings) == 0:
        print("No face embeddings found in the database.")
        return

    print(f"Clustering {len(embeddings)} total faces using DBSCAN...")

    # Run DBSCAN with cosine metric (ideal for normalized face vectors)
    # eps: distance threshold (lower = stricter matching, higher = looser grouping)
    clt = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    clt.fit(embeddings)

    labels: np.ndarray = clt.labels_
    unique_labels: list[int] = sorted(set(labels))

    num_people: int = len([l for l in unique_labels if l != -1])
    noise_count: int = int(np.sum(labels == -1))

    print("\n--- Clustering Complete ---")
    print(f"Found {num_people} unique person clusters.")
    print(f"Unmatched / Noise faces (-1): {noise_count}\n")

    # Group and display results by cluster label
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise/unmatched singles for the main person list

        # Get all indices belonging to this person cluster
        indices: np.ndarray = np.where(labels == label)[0]
        print(f"👤 Person #{label} ({len(indices)} images):")

        for idx in indices:
            path: str = str(image_paths[idx])
            box: np.ndarray = bboxes[idx]
            print(f"  -> File: {path}")
            print(f"     Face BBox: {box.tolist()}")


onnxruntime.preload_dlls()




def process_image_folder(folder_path, use_gpu=False):
    name = "antelopev2"
    print(f"Initializing InsightFace ({name})...")
    ctx_id = 0 if use_gpu else -1  # 0 for CUDA/GPU, -1 for CPU-only execution
    app = insightface.app.FaceAnalysis(name=name)  # buffalo_l
    app.prepare(ctx_id=ctx_id, det_size=(1280, 1280))

    target_dir = pathlib.Path(folder_path)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory '{folder_path}' does not exist or is not a folder.")
        return

    # Recursively find all image files
    image_files = [
        p for p in target_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", "png")
    ]
    print(f"Found {len(image_files)} image files in '{target_dir}'. Starting scan...\n")

    image_paths: list[str] = []
    bboxes: list[list[float]] = []
    embeddings: list[list[float]] = []

    nofaces_paths: list[str] = []

    image_files_len = len(image_files)
    interval = max(1, image_files_len // 20)

    print(f"Progress: 0% (0/{image_files_len})")
    for i, img_path in enumerate(image_files):
        
        if (i + 1) % interval == 0 or (i + 1) == image_files_len:
            percent = int(((i + 1) / image_files_len) * 100)
            print(f"Progress: {percent}% ({i + 1}/{image_files_len})")
        
        # print(f"Scanning: {img_path.name}")
        try:
            # Read image with OpenCV
            image_bytes = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if img is None:
                print(f"  -> Warning: Failed to read image (file may be corrupt). {str(img_path)}")
                continue

            # Detect faces and extract embeddings
            faces = app.get(img)
            # print(f"  -> Detected {len(faces)} face(s).")
            if len(faces) == 0:
                nofaces_paths.append(str(img_path))
            else:
                for face in faces:
                    image_paths.append(str(img_path))
                    bboxes.append(face.bbox.tolist())  # [x1, y1, x2, y2]
                    embeddings.append(face.embedding.tolist())  # 512-dim vector

        except Exception as e:  # noqa: BLE001
            print(f"  -> Error during processing: {e}")

    # Save efficiently into a compressed binary NumPy archive (.npz)

    output_filename: str = f"face_embeddings_{target_dir.name}.npz"
    print(f"\nSaving binary face database to {output_filename}...")


    np.savez_compressed(
        output_filename,
        paths=np.array(image_paths),
        bboxes=np.array(bboxes, dtype=np.float32),
        embeddings=np.array(embeddings, dtype=np.float32),
    )


    np.savez_compressed(
        f"noface_images_paths_{target_dir.name}",
        paths=np.array(nofaces_paths),
    )

    print(
        f"Processing complete! Saved {len(embeddings)} total faces across "
        f"~{len(image_paths)} images into {output_filename}."
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <directory>")
        return
    
    # Change this to the path of your photo folder
    target_folder = sys.argv[1]


    process_image_folder(target_folder, use_gpu=True)
    # cluster_faces(npz_path="face_embeddings.npz", eps=0.4, min_samples=2)

if __name__ == "__main__":
    main()
