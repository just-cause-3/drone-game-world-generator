# Stage 1 model files

Run this once from the repository root:

```powershell
.venv\Scripts\python.exe setup_drone_model.py
```

The setup command downloads the pinned `dronefreak/visdrone-yolov26x` checkpoint,
checks its immutable revision, byte size, SHA-256, and class names, then
installs it as `models/visdrone-yolo26x.pt`. Model weights are intentionally ignored
by Git.

The checkpoint contains the ten documented VisDrone classes plus an `others` class.
Stage 1 loads the full checkpoint but its current project policy masks only pedestrians
and cars. Every report records the complete model class map, selected subset, and weight
hash.

`train_drone_model.py` is the reproducible fallback for training from the official
Ultralytics `VisDrone.yaml`. Keep a newly trained model under a different name until
it has been evaluated against the pinned baseline and representative project footage.
