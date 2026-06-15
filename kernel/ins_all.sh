#!/usr/bin/env bash

echo '--- Loading benchmark TCP congestion-control modules ---'

modules=(
  tcp_bbr
  tcp_bbr1
  tcp_westwood
  tcp_veno
  tcp_vegas
  tcp_yeah
  tcp_cdg
  tcp_bic
  tcp_htcp
  tcp_hybla
  tcp_highspeed
  tcp_illinois
)

for module in "${modules[@]}"; do
  if sudo modprobe "$module"; then
    echo "loaded $module"
  else
    echo "warning: could not load $module" >&2
  fi
done

echo '--- Available TCP congestion controls ---'
cat /proc/sys/net/ipv4/tcp_available_congestion_control

# Load these public algorithms before inserting the custom astraea or mutant
# modules, because mutant references their exported kernel structures.
