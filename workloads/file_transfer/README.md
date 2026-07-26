# File-transfer flow-completion-time workload

This workload launches staggered, concurrent `curl` downloads from the Windows
client (`phoenix`, Tailscale `100.118.38.15`) to the Linux server (`mihai`,
Tailscale `100.90.202.72`). It follows the direct Tailscale path used by the
Jellyfin workload; no SSH port forwarding is involved.

`curl` discards each response body but records the full network transfer.
Every flow gets a status, transferred byte count, mean goodput, time to first
byte, and flow completion time (FCT). `-Duration` is a per-flow timeout, not a
fixed traffic duration. A timed-out transfer has a blank FCT and remains in the
dataset as an incomplete observation.

## 1. Prepare the Linux server

If the repository is not already on `mihai`, copy the two server scripts from
this directory in Windows PowerShell (replace the username):

```powershell
$LinuxUser = "your-linux-user"
ssh "${LinuxUser}@100.90.202.72" "mkdir -p ~/olympus-file-transfer"
scp .\setup_file_server.sh .\serve_file_server.sh `
  "${LinuxUser}@100.90.202.72:~/olympus-file-transfer/"
```

Then, on `mihai`, run:

```bash
cd ~/olympus-file-transfer
chmod +x setup_file_server.sh serve_file_server.sh
sudo ./setup_file_server.sh
```

The setup creates these uncompressed test objects:

```text
/srv/olympus-file-transfer/file-100MiB.bin
/srv/olympus-file-transfer/file-500MiB.bin
/srv/olympus-file-transfer/file-1GiB.bin
```

It also permits TCP port 8080 on `tailscale0` when UFW is active. The server
binds only to `100.90.202.72`, so it is not exposed on the public interface.
For a different address, port, or directory, pass environment variables to
the setup:

```bash
sudo env TAILSCALE_IP=100.90.202.72 FILE_PORT=8080 \
  FILE_ROOT=/srv/olympus-file-transfer ./setup_file_server.sh
```

The generated files are zero-filled by design. The HTTP server does not
compress responses, so their full sizes cross the network. Reuse exactly the
same file set for every approach.

## 2. Select the actual congestion-control approach

The Linux box is the TCP sender, so configure the approach there before each
batch. For an in-kernel algorithm:

```bash
cat /proc/sys/net/ipv4/tcp_available_congestion_control
sudo sysctl -w net.ipv4.tcp_congestion_control=cubic
sysctl net.ipv4.tcp_congestion_control
```

Replace `cubic` with the available kernel algorithm being tested. For
Orca/Astraea or another controller-driven approach, start its listener and
model using that approach's normal deployment procedure before launching the
downloads.

Start the file server only after selecting the approach:

```bash
sudo ./serve_file_server.sh
```

For a non-default address, port, or directory:

```bash
sudo env TAILSCALE_IP=100.90.202.72 FILE_PORT=8080 \
  FILE_ROOT=/srv/olympus-file-transfer ./serve_file_server.sh
```

Leave it running during the batch. Before testing another approach, stop the
server with Ctrl+C, change the approach, and start the server again. Restarting
matters because accepted connections may inherit congestion control from the
listening socket that was created at server startup.

From Windows, check the direct route before measuring:

```powershell
curl.exe --noproxy "*" -I http://100.90.202.72:8080/file-100MiB.bin
```

The PowerShell `-CC` value is deliberately only a data label. It does not
change Linux congestion control. Make it match the method you actually
configured. While transfers are active, verify the server sockets and peer:

```bash
ss -tinp 'sport = :8080'
```

The peer should be `100.118.38.15`, not `127.0.0.1`.

## 3. Run transfers from Windows

In this directory:

```powershell
.\run_n_transfers.ps1 -N 5 -Duration 100 -Stagger 20 -CC orca
```

This starts flow 1 immediately and one further flow every 20 seconds. Each
flow may run for up to 100 seconds, so the whole command can take approximately
`(N - 1) * Stagger + Duration` seconds.

The default object is `file-100MiB.bin`. Choose one larger object:

```powershell
.\run_n_transfers.ps1 -N 5 -Duration 300 -Stagger 20 -CC cubic `
  -Files file-500MiB.bin
```

Or rotate several objects across the N flows:

```powershell
.\run_n_transfers.ps1 -N 6 -Duration 300 -Stagger 10 -CC bbr `
  -Files file-100MiB.bin,file-500MiB.bin,file-1GiB.bin
```

Keep `N`, `Duration`, `Stagger`, and `Files` identical when comparing
approaches. Results are written to:

```text
data/<cc>/n<N>-transfer-<datetime>-<id>/
  metadata.json
  summary.csv
  logs/
    transfer-1.curl.json
    transfer-1.err.log
    ...
```

## 4. Repeat and average

Run a complete batch several times under each approach. This example performs
five repetitions:

```powershell
1..5 | ForEach-Object {
  .\run_n_transfers.ps1 -N 5 -Duration 100 -Stagger 20 -CC orca
}
```

Then combine every run and calculate averages:

```powershell
python .\analyze_results.py
```

The analysis produces:

```text
analysis/combined_flows.csv
analysis/fct_averages.csv
```

`fct_averages.csv` contains an `all_flows` row and repeated-run averages for
each stagger position. It reports completion rate, mean/stdev/median/p95 FCT,
and mean goodput. FCT statistics use completed transfers only; always interpret
them together with `completion_rate` and `timeout_count`, because excluding
timeouts alone can make an approach look artificially fast.
