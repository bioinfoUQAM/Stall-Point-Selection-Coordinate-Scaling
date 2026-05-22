"""
select_check_points.py
-----------------------
Interactive tool to select reference points (e.g., stall corners) on the first
frame of a fixed-camera video, then normalise bounding box coordinates relative
to the stall geometry for cross-video HMM feature comparison.

Usage:
    python select_check_points.py --video <path> --input <path> --output <path> [options]

Arguments:
    --video             Path to the .mp4 video file.
    --input             Path to the input feature table (.csv or .xlsx).
    --output            Path for the output scaled CSV.
    --prefixes          Body-part column prefixes to normalise (default: "body box" "head box" "snout box").
    --display_width     Display window width in pixels (default: 1200).
    --display_height    Display window height in pixels (default: 800).
    --no_click          Skip interactive selection and use --stall_points instead.
    --stall_points      8 integers: x1 y1 x2 y2 x3 y3 x4 y4 (LT LB RB RT in pixels).

Controls (interactive window):
    Left click      Add a point.
    Right click     Remove the last point.
    S               Save points to JSON.
    C               Clear all points.
    R               Reset view.
    ESC             Finish selection.
"""

import argparse
import os
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_table_any(path):
    """Read a .csv or .xlsx file into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def stall_center_and_size_from_4pts(stall_pts_px):
    """
    Compute stall center, width, and height from 4 corner points.

    Parameters
    ----------
    stall_pts_px : list of 4 (x, y) tuples
        Corners in order: LT, LB, RB, RT.

    Returns
    -------
    cx, cy, ws, hs : float
        Center coordinates, stall width, stall height (all in pixels).
    """
    pts = np.asarray(stall_pts_px, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError("Exactly 4 stall points required in order LT, LB, RB, RT.")

    LT, LB, RB, RT = pts

    cx = 0.5 * (LT[0] + RB[0])
    cy = 0.5 * (LT[1] + RB[1])

    def dist(a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    ws = 0.5 * (dist(LT, RT) + dist(LB, RB))
    hs = 0.5 * (dist(LT, LB) + dist(RT, RB))

    return cx, cy, ws, hs


def normalize_tlwh(df, prefix, cx, cy, ws, hs):
    """
    Normalise bounding box columns for a given body-part prefix relative to
    the stall center and dimensions. Adds Ln, Tn, Wn, Hn and aspect ratio columns.

    Parameters
    ----------
    df : pd.DataFrame
    prefix : str
        Column prefix, e.g. "body box". Expects columns "<prefix> L/T/W/H".
    cx, cy : float
        Stall center in pixels.
    ws, hs : float
        Stall width and height in pixels.

    Returns
    -------
    pd.DataFrame with additional normalised columns.
    """
    Lc, Tc, Wc, Hc = f"{prefix} L", f"{prefix} T", f"{prefix} W", f"{prefix} H"
    for c in [Lc, Tc, Wc, Hc]:
        if c not in df.columns:
            raise KeyError(f"Missing column: '{c}'. Check --prefixes argument.")

    L = df[Lc].astype(float).to_numpy()
    T = df[Tc].astype(float).to_numpy()
    W = df[Wc].astype(float).to_numpy()
    H = df[Hc].astype(float).to_numpy()

    df[f"{prefix} Ln"] = (L - cx) / (ws / 2.0)
    df[f"{prefix} Tn"] = (T - cy) / (hs / 2.0)
    df[f"{prefix} Wn"] = W / ws
    df[f"{prefix} Hn"] = H / hs

    AR       = W / (H + 1e-9)
    AR_stall = (W / ws) / ((H / hs) + 1e-12)

    df[f"{prefix} AR"]           = AR
    df[f"{prefix} AR_stall"]     = AR_stall
    df[f"{prefix} logAR"]        = np.log(AR)
    df[f"{prefix} logAR_stall"]  = np.log(AR_stall + 1e-12)

    return df


# ---------------------------------------------------------------------------
# Interactive point selector
# ---------------------------------------------------------------------------

class VideoPointSelector:
    """
    Opens the first frame of a video in an interactive window and lets the user
    click reference points (e.g., stall corners). Points are stored in both
    display and original frame coordinates.
    """

    def __init__(self, video_path, display_width=1200, display_height=800):
        self.video_path     = video_path
        self.display_width  = display_width
        self.display_height = display_height

        self.selected_points = []   # display-space coordinates
        self.original_points = []   # original frame coordinates

        self.original_frame = None
        self.display_frame  = None
        self.working_frame  = None

        self.scale_x = 1.0
        self.scale_y = 1.0
        self.window_name = "Select Points  |  Left: Add  |  Right: Remove  |  ESC: Done"

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def load_first_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video '{self.video_path}'.")
            return False

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("ERROR: Cannot read the first frame.")
            return False

        self.original_frame = frame.copy()
        h, w = frame.shape[:2]
        print(f"Original frame: {w}x{h} px")

        scale = min(self.display_width / w, self.display_height / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        self.scale_x = self.scale_y = scale

        self.display_frame = cv2.resize(frame, (new_w, new_h))
        self.working_frame = self.display_frame.copy()
        print(f"Display frame:  {new_w}x{new_h} px  (scale={scale:.3f})")
        return True

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_point(self, x, y, number):
        cv2.circle(self.working_frame, (x, y), 10, (0, 255, 0), -1)
        cv2.circle(self.working_frame, (x, y), 10, (0, 0, 0),   2)
        cv2.putText(self.working_frame, str(number), (x - 6, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        coord_text = f"({x},{y})"
        tx, ty = x + 15, y - 15
        fh, fw = self.working_frame.shape[:2]
        if tx + 80 > fw: tx = x - 100
        if ty < 20:      ty = y + 35

        ts = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(self.working_frame,
                      (tx - 2, ty - ts[1] - 2), (tx + ts[0] + 2, ty + 2),
                      (0, 0, 0), -1)
        cv2.putText(self.working_frame, coord_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _redraw_all(self):
        self.working_frame = self.display_frame.copy()
        for i, (x, y) in enumerate(self.selected_points):
            self._draw_point(x, y, i + 1)
        cv2.imshow(self.window_name, self.working_frame)

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append((x, y))
            ox, oy = int(x / self.scale_x), int(y / self.scale_y)
            self.original_points.append((ox, oy))
            self._draw_point(x, y, len(self.selected_points))
            cv2.imshow(self.window_name, self.working_frame)
            print(f"Point {len(self.selected_points)}: display({x},{y}) -> original({ox},{oy})")

        elif event == cv2.EVENT_RBUTTONDOWN and self.selected_points:
            rd, ro = self.selected_points.pop(), self.original_points.pop()
            self._redraw_all()
            print(f"Removed: display{rd} -> original{ro}  |  Remaining: {len(self.selected_points)}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def clear_points(self):
        self.selected_points.clear()
        self.original_points.clear()
        self._redraw_all()
        print("All points cleared.")

    def reset_view(self):
        self._redraw_all()
        print("View reset.")

    def save_coordinates(self, output_dir="."):
        if not self.original_points:
            print("No points to save.")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"selected_coordinates_{ts}.json")
        data = {
            "video_file":        self.video_path,
            "timestamp":         ts,
            "total_points":      len(self.original_points),
            "original_frame_size": {
                "width":  self.original_frame.shape[1],
                "height": self.original_frame.shape[0]
            },
            "coordinates": {
                "original_frame": self.original_points,
                "display_frame":  self.selected_points
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {path}")

    def display_summary(self):
        print(f"\n{'='*50}")
        print(f"SUMMARY — {len(self.original_points)} point(s) selected")
        print(f"{'='*50}")
        print(f"{'#':<4} {'Original (px)':<18}")
        print("-" * 30)
        for i, (ox, oy) in enumerate(self.original_points):
            print(f"{i+1:<4} ({ox:4d}, {oy:4d})")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_selection(self, save_dir="."):
        if not self.load_first_frame():
            return []

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window_name, self.display_frame.shape[1], self.display_frame.shape[0])
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.imshow(self.window_name, self.working_frame)

        print("\n" + "="*60)
        print("  Left click : Add point")
        print("  Right click: Remove last point")
        print("  S          : Save points to JSON")
        print("  C          : Clear all points")
        print("  R          : Reset view")
        print("  ESC        : Confirm and close")
        print("="*60)

        while True:
            key = cv2.waitKey(30) & 0xFF
            if   key == 27:                         break
            elif key in (ord("s"), ord("S")):       self.save_coordinates(save_dir)
            elif key in (ord("c"), ord("C")):       self.clear_points()
            elif key in (ord("r"), ord("R")):       self.reset_view()

        cv2.destroyAllWindows()
        self.display_summary()
        return self.original_points


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select stall corners and normalise bounding box coordinates.")
    parser.add_argument("--video",          required=True,
                        help="Path to the .mp4 video file.")
    parser.add_argument("--input",          required=True,
                        help="Path to the input feature table (.csv or .xlsx).")
    parser.add_argument("--output",         required=True,
                        help="Path for the output scaled CSV.")
    parser.add_argument("--prefixes",       nargs="+",
                        default=["body box", "head box", "snout box"],
                        help="Body-part column prefixes to normalise.")
    parser.add_argument("--display_width",  type=int, default=1200,
                        help="Display window width (default: 1200).")
    parser.add_argument("--display_height", type=int, default=800,
                        help="Display window height (default: 800).")
    parser.add_argument("--no_click",       action="store_true",
                        help="Skip interactive selection; use --stall_points instead.")
    parser.add_argument("--stall_points",   type=int, nargs=8,
                        metavar=("x1","y1","x2","y2","x3","y3","x4","y4"),
                        help="Hardcoded stall corners in pixels: LT LB RB RT.")
    args = parser.parse_args()

    # Step 1  get stall points
    if args.no_click:
        if args.stall_points is None:
            raise SystemExit("--stall_points required when --no_click is set.")
        coords = list(zip(args.stall_points[0::2], args.stall_points[1::2]))
        stall_pts_px = coords
        print("Using hardcoded stall points (LT, LB, RB, RT):", stall_pts_px)
    else:
        selector = VideoPointSelector(args.video,
                                      display_width=args.display_width,
                                      display_height=args.display_height)
        save_dir = os.path.dirname(args.output) or "."
        stall_pts_px = selector.run_selection(save_dir=save_dir)
        if len(stall_pts_px) != 4:
            raise SystemExit(f"Need exactly 4 stall points. Got {len(stall_pts_px)}.")

    # Step 2  compute stall geometry
    cx, cy, ws, hs = stall_center_and_size_from_4pts(stall_pts_px)
    print(f"Stall center: ({cx:.1f}, {cy:.1f}) | width: {ws:.1f} px | height: {hs:.1f} px")

    stall_pts_norm = (np.asarray(stall_pts_px, dtype=np.float64)
                      - np.array([cx, cy])) / np.array([ws / 2.0, hs / 2.0])
    print("Stall corners (normalised):", [tuple(np.round(p, 4)) for p in stall_pts_norm.tolist()])

    # Step 3  load feature table
    df = read_table_any(args.input)
    print(f"Loaded table: {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 4  normalise bounding boxes
    for prefix in args.prefixes:
        df = normalize_tlwh(df, prefix, cx, cy, ws, hs)
        print(f"Normalised: {prefix}")

    # Step 5  export
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
