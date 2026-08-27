# Dataset archive

Raw dataset zips, chunked into <100MB pieces (GitHub's hard per-file limit)
and committed directly to this repo, so any machine can `git clone` /
`git pull` and get everything needed without re-downloading from CIC or AWS.

Reassemble + extract with:

```bash
bash scripts/reassemble_datasets.sh            # all available datasets
bash scripts/reassemble_datasets.sh cicids2017 # just one
```

This cats the chunks back into the original zip, verifies it against
`checksums.txt`, and extracts into `data_raw/<dataset>/` (gitignored --
that's where `armor_fl.data.preprocessing` reads from).

## Provenance

| Dataset | Source | Archived as | Size |
|---|---|---|---|
| CICIDS2017 | Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/ids-2017.html (download portal: cicresearch.ca), file `MachineLearningCSV.zip` | `MachineLearningCSV_CICIDS2017.zip` | 224 MB |
| CICIDS2018 | https://www.unb.ca/cic/datasets/ids-2018.html -- `aws s3 sync --no-sign-request s3://cse-cic-ids2018/"Processed Traffic Data for ML Algorithms/" <dest>` (only the processed ML CSVs, not the raw pcap/log half of the bucket) | `CICIDS2018_ProcessedTrafficData.tar.gz` (re-archived here as tar.gz -- the S3 source is loose CSVs, not a zip) | 1.6 GB (gzipped from 6.5 GB) |
| CICIoT2023 | Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/iotdataset-2023.html, file `MERGED_CSV.zip` | `MERGED_CSV_CICIoT2023.zip` | 1.6 GB |

Loaders for all three are in `armor_fl.data.preprocessing`: `load_cicids2017`,
`load_cicids2018`, `load_ciciot2023`.

`checksums.txt` holds the SHA-256 of each *original* (unsplit) archive, used
to verify reassembly succeeded byte-for-byte.
