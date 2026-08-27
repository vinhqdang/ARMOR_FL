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

| Dataset | Source | Original file | Size |
|---|---|---|---|
| CICIDS2017 | Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/ids-2017.html (download portal: cicresearch.ca) | `MachineLearningCSV.zip` | 224 MB |
| CICIDS2018 | https://www.unb.ca/cic/datasets/ids-2018.html -- `aws s3 sync --no-sign-request s3://cse-cic-ids2018/ <dest>` | (S3 bucket, not yet archived here) | TBD |
| CICIoT2023 | Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/iotdataset-2023.html | `MERGED_CSV.zip` | 1.6 GB |

CICIDS2018 is not yet chunked into this archive (still being downloaded /
its full size isn't confirmed) -- once available, extend
`scripts/reassemble_datasets.sh` and this table the same way.

`checksums.txt` holds the SHA-256 of each *original* (unsplit) zip, used to
verify reassembly succeeded byte-for-byte.
