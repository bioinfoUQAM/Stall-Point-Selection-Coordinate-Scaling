# Stall Point Selection & Coordinate Scaling

Tool to interactively select the 4 corners of an animal stall on the first video frame, then normalize bounding box coordinates relative to the stall geometry. Part of the [WELL-E](https://github.com/bioinfoUQAM) animal welfare computer vision pipeline.

---

## Features

- Interactive point selection directly on the first video frame.
- Computes stall center, width, and height from 4 clicked corner points.
- Normalises body, head, and snout bounding boxes to a stall-centered coordinate system.
- Adds aspect ratio features (`AR`, `AR_stall`, `logAR`, `logAR_stall`) per body part.
- Supports both interactive and hardcoded stall point input.
- Reads `.csv` and `.xlsx` input tables.

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- pandas
- openpyxl *(only for `.xlsx` inputs)*

```bash
pip install opencv-python numpy pandas openpyxl
```

> **Note:** An interactive display (desktop, X forwarding, or VNC) is required to run the point selector.

---

## Usage

```bash
python select_check_points.py
```

Edit the paths at the bottom of the script before running:

```python
VIDEO_PATH  = "path/to/video.mp4"
INPUT_TABLE = "path/to/features.csv"    # .csv or .xlsx
OUTPUT_CSV  = "path/to/output_scaled.csv"
CLICK_STALL_POINTS = True               # False to use hardcoded points
```

To skip the interactive window, set `CLICK_STALL_POINTS = False` and provide:

```python
STALL_POINTS_PX = [
    (177,  410),   # LT — Left-Top
    (189, 1231),   # LB — Left-Bottom
    (924, 1212),   # RB — Right-Bottom
    (931,  396),   # RT — Right-Top
]
```

---

## Point Order

Click the 4 stall corners in this exact order:
LT ───── RT
│         │
│  STALL  │
│         │
LB ───── RB

Exactly **4 points** must be selected the script will raise an error otherwise.

---

## Controls

| Action | Control |
|--------|---------|
| Add point | Left click |
| Remove last point | Right click |
| Save points to JSON | `S` |
| Clear all points | `C` |
| Reset view | `R` |
| Finish | `ESC` |

---

## Inputs / Outputs

| | File | Description |
|--|------|-------------|
| **Input** | `.mp4` video | Fixed-camera recording first frame is used |
| **Input** | `features.csv` / `.xlsx` | Bounding box feature table (output of `inter_filt.py`) |
| **Output** | `*_scaled.csv` | Same table with added normalised columns |
| **Output** *(optional)* | `selected_coordinates_<timestamp>.json` | Saved clicked points in pixel and display coordinates |

### Added columns (per body part)

Applied to prefixes `body box`, `head box`, `snout box`:

| Column | Description |
|--------|-------------|
| `Ln`, `Tn` | Position normalised to stall center and half-dimensions |
| `Wn`, `Hn` | Size normalised by stall width and height |
| `AR` | Raw aspect ratio W/H |
| `AR_stall` | Aspect ratio normalised by stall dimensions |
| `logAR`, `logAR_stall` | Natural log of the above |

---

## Pipeline
rearrange_boxes.py → *_arranged_boxes_.csv
assign_boxes.py → roi_<id>_new.csv
inter_filt.py → _interpolated_filtered_derivates_.csv
select_check_points.py → *_scaled.csv
│
▼
HMM analysis

---

## Notes

- Fixed-camera setup required stall corners must be stable in frame 1.
- Scaling is 2D only; camera tilt or perspective distortion is not corrected.
- Hardcoded stall points can be reused across runs for reproducibility.

---

*Part of the WELL-E animal welfare research pipeline — UQAM / McGill.*
