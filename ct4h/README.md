# CT4Harmonization-Multicentric: low-dose to high-dose translation

Second image-translation testbed after MRIxFields. Data: TCIA CT4Harmonization-Multicentric
(doi:10.7937/M0PB-BH69): one iodine-ink anthropomorphic phantom (thoracic, liver with six ROIs, test patterns)
scanned on 13 scanners, 5 doses (1, 3, 6, 10, 14 mGy), 10 repeats; FBP, IR and (D1, E2) DL reconstructions.

## What goes where

| use | data |
|---|---|
| training / paired evaluation | the dataset authors' registered volumes (below) |
| NPS, power spectra, error spectra | raw DICOM pixels (unclipped HU, no interpolation). Their z resampling and B-spline interpolation low-pass the noise, so spectra of the registered volumes are not the scanner's. Measuring that change is a result in itself. |

All metrics use `ct4h.data.utils.HU_RANGE = (-1000, 2000)`, the range the authors' code keeps.

## Index

Every tool starts from `ct4h_index.csv`, one row per CT series (scanner, dose_mgy, recon, repeat, series_dir,
seg_dir for the liver ROI masks, n_slices, pixel_spacing_mm, slice_thickness_mm ...). Two steps build it from the
TCIA download (they need pydicom and pandas):

```bash
python scripts/scan_ct4h_headers.py <DICOM root> <ct4h_series_index.csv>     # headers only, one row per series
python scripts/build_ct4h_index.py  <ct4h_series_index.csv> <ct4h_index.csv>
```

`<DICOM root>` is the folder that holds `CT4Harmonization-Multicentric/<study>/...` (the same folder is `--data_dir`
and `CT4H_DIR` later). The index builder stops if a series description does not parse, if the scanner tag disagrees
with the study, or if a manufacturer is unknown, and it lists series with fewer slices than their scanner's usual
count.

## Registered volumes

`scripts/run_authors_registration.py` runs the loop body of the authors' `registered_dataset.py`
(Harmonization-Dataset, medgift) on every series: read DICOM, flip decision, circular z roll, rigid elastix to one
reference (A1, 10 mGy, IR, repeat 1), B-spline resampling. It calls their functions from their clone; the algorithm
is unchanged apart from the options below. It needs a separate environment with itk-elastix, antspyx, dcmstack, monai,
nibabel and a CUDA torch (kept apart from the training env).

```bash
python scripts/run_authors_registration.py --index <ct4h_index.csv> --data_dir <DICOM root> \
    --authors_repo <Harmonization-Dataset clone> --out_dir <authors_registered> \
    --shard K/N                                # N shards in parallel (--fallback_rule: see below)
python scripts/make_ct4h_manifest.py --reg_dir <authors_registered>
```

Output: `<scanner>_<dose>mGy_<recon>_rep<NN>.nii.gz` (float32, shape z x y x x = 310 x 275 x 381) and
`registration_manifest.csv`. Read the manifest, not the folder.

The authors' own script registers every reconstruction (`reconstruction_method = '*'`). We registered FBP first
(`--recons FBP`, the default), then IR and DL with `--roll_from`, so that the three reconstructions of one scan share
its FBP roll.

Options added on top of the authors' code:

- `--fallback_rule`: their rule drops a good registration (it compares the whole-volume SSIM before with the cropped
  SSIM after; 140 of 649 scans hit it). The default here, `like_for_like`, compares cropped with cropped; 32 scans hit
  it in the first run, and A1 (the reference's own scanner) was then rerun with `never`. In the final set
  `fallback = 0` for all 648 volumes, i.e. every file is a registered volume. Both rules are logged either way.
- `--force_roll SCANNER=SLICES`: their roll search only tries multiples of 4 slices. On noisy low-dose F1 scans it
  flipped between 4 and 8 (15 of 50 scans got 8), which left a constant 1.28 mm (one native slice) z offset. Those
  15 were rerun with `--force_roll F1=4`, the value the other 35 chose on their own. The roll is constant per
  scanner for 11 of 13 scanners (A1 0, A2 0, B1 4, B2 0, C1 0, D1 12, E1 8, G1 0, G2 0, H1 0, H2 4).

- `--roll_from <registration_manifest.csv>`: each series takes the roll its FBP twin got (same scan, same grid;
  looked up by scanner, dose and repeat). Used for `--recons IR DL`, because the roll search is noisy and would
  otherwise be free to differ between reconstructions of one acquisition. Logged as `roll_forced = 2`.

## Checked and known

- Registered volumes agree with the reference to <= 0.22 mm in plane and <= 0.4 mm in z (profile matching,
  sub-slice) on all 648 scans, after the F1 fix. The phantom's z structure is weak inside the liver and test-pattern
  segments, so z is determined mainly by the spine and the segment ends.
- E2: roll is mixed (20 scans at 0, 30 at 4), and about half of the 6 and 10 mGy scans (and all 14 mGy scans) show a
  y_high band offset of 0.5 to 1.1 mm in z against the A1 reference. Matched pairs differ by <= ~0.75 mm. Accepted
  as a limitation; revisit only if E2 is worse than the other scanners in the per-scanner results.
- B2 14 mGy repeat 5 has no registered volume: its DICOM stack has missing slices (dcmstack: inconsistent slice
  spacing). Its pairs are excluded. E1 has 49 scans in the dataset, so the set is 648 volumes.
- The alignment and z-profile diagnostics used to establish this are tagged `ct4h-diagnostics-archive` in git.

## Layout

| path | what |
|---|---|
| `ct4h/data/utils.py` | doses, recons, scanners, vendors, `HU_RANGE`, domain ids, index reading |
| `ct4h/env.py` | `CT4H_DIR`, `CT4H_PREPROCESSED_DIR` from `.env` |
| `scripts/scan_ct4h_headers.py` | DICOM headers to one row per series |
| `scripts/build_ct4h_index.py` | header scan to `ct4h_index.csv` |
| `scripts/run_authors_registration.py` | the authors' registration over our index |
| `scripts/make_ct4h_manifest.py` | newest log row per series, only for files on disk |

Cite the dataset (doi:10.7937/M0PB-BH69) and the authors' paper (arXiv 2507.01539) in any publication.
