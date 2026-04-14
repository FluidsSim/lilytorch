"""
Script to list video files in a folder and record their creation date/time
(read from companion XML metadata files) into a summary CSV.
"""

import os
import datetime
import xml.etree.ElementTree as ET
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────
MAIN_DIR = "/data/andreaferrario/1guilla_experiments/dyes"
VIDEO_DIR = MAIN_DIR + "/original videos"
SAVE_DIR = MAIN_DIR + "/log/kinematic_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

NS = {"nrt": "urn:schemas-professionalDisc:nonRealTimeMeta:ver.2.00"}


def read_creation_date_from_xml(xml_path):
    """Parse the CreationDate value from a Sony-style XML sidecar file
    and return a local datetime string with the UTC offset applied."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    elem = root.find("nrt:CreationDate", NS)
    if elem is not None:
        raw = elem.attrib["value"]  # e.g. "2025-09-02T11:40:52+02:00"
        dt = datetime.datetime.fromisoformat(raw)
        # Apply the offset to get absolute local time, then drop tzinfo
        dt_local = (dt).replace(tzinfo=None)
        return dt_local.strftime("%Y-%m-%d %H:%M:%S")
    return None


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    video_files = sorted(
        f for f in os.listdir(VIDEO_DIR)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    )
    print(f"Found {len(video_files)} video files in {VIDEO_DIR}\n")

    rows = []
    for fname in video_files:
        # Derive the companion XML filename: e.g. C0070.MP4 → C0070M01.XML
        stem = os.path.splitext(fname)[0]
        xml_name = f"{stem}M01.XML"
        xml_path = os.path.join(VIDEO_DIR, xml_name)

        if os.path.isfile(xml_path):
            dt_str = read_creation_date_from_xml(xml_path)
        else:
            dt_str = "XML not found"
            print(f"  WARNING: {xml_name} not found for {fname}")

        rows.append({"filename": fname, "creation_datetime": dt_str})
        print(f"  {fname}  →  {dt_str}")

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(SAVE_DIR, "video_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary CSV saved → {summary_path}")
    print("Done.")
