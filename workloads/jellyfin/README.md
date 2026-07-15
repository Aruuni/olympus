# Jellyfin client-side benchmark on Linux

## Install

```bash
mkdir -p ~/jellyfin-benchmark
cd ~/jellyfin-benchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
playwright install --with-deps chromium
```

## Save a login session

```bash
playwright codegen --save-storage=auth.json \
  "https://jellyfin.phoenixremoteaccess.uk/web/"
```

Log in, verify the item is accessible, then close the Playwright browser.

Protect the saved token:

```bash
chmod 600 auth.json
printf 'auth.json\nresults/\n.venv/\n' >> .gitignore
```

## First run

```bash
chmod +x run_benchmark.sh
./run_benchmark.sh baseline-linux
```

Click Play when prompted. Results are written to `results/<run-id>/`.

## Automatic play

After confirming the UI works, run directly without `--manual-play`:

```bash
.venv/bin/python jellyfin_client_benchmark.py \
  --url "https://jellyfin.phoenixremoteaccess.uk/web/#/details?id=762277cc91effea4c30f648c3c106797&serverId=676fc56ba97448b3a598c30f8be4a1ce" \
  --auth auth.json \
  --duration 180 \
  --label baseline-linux
```

## Repeat ten times

```bash
for i in $(seq 1 10); do
  .venv/bin/python jellyfin_client_benchmark.py \
    --url "https://jellyfin.phoenixremoteaccess.uk/web/#/details?id=762277cc91effea4c30f648c3c106797&serverId=676fc56ba97448b3a598c30f8be4a1ce" \
    --auth auth.json \
    --duration 180 \
    --label "baseline-linux-rep-$i"
done
```

## Combine run summaries

```bash
.venv/bin/python combine_summaries.py \
  --results results \
  --output all-runs.csv
```
