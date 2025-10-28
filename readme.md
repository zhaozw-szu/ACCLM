# Adaptive Container Characterization via Language Mapping for Video Integrity and Source Analysis

This is the official repository of *Adaptive Container Characterization via Language Mapping for Video Integrity and Source Analysis*.

---

## Dependencies

### Python

Install required Python packages:

```bash
pip install -r requirements.txt
```

Ensure you are using **Python 3.8 or higher**.

### JavaScript / Node.js

This project uses `mp4box` for low-level MP4 container parsing.

- **mp4box**: `0.5.2`  
  A JavaScript library for parsing ISO Base Media Format (e.g., MP4) files.  
  GitHub: [https://github.com/gpac/mp4box.js](https://github.com/gpac/mp4box.js)

> The exact version is locked in `package-lock.json`.


Weight and traininng code,   will be released soon.


## Extract Container Information

### 1. Extract MP4 Container Structure

Parse MP4 files listed in a CSV and save the container structure as JSON:

```bash
python get_container/get_container.py --csv_path "mp4.csv" --save_path "container_json/mp4.json"
```

- `mp4.csv`: A CSV file containing video paths and labels for all MP4-based format videos(Supported formats mentioned: `.mp4`, `.mov`, `.m4v`, `.3gp`, etc.).
- `container_json/mp4.json`: Output file storing parsed container trees.

### 2. Extract AVI Container Structure (Optional)

For AVI format videos:

```bash
python get_container/avi_parse.py --csv_path "avi.csv" --save_path "container_json/avi.json"
```

### 3. Extract MTS Container Structure (Optional)

For MTS format videos:

```bash
python get_container/mts_parse.py --csv_path "mts.csv" --save_path "container_json/mts.json"
```



## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{
  title={Adaptive Container Characterization via Language Mapping for Video Integrity and Source Analysis},
  journal={in peer review},
}
```

Replace the above BibTeX entry with your final publication details.

