#!/bin/bash


echo '--- Inserting protocols as modules ---'

sudo modprobe tcp_bbr
sudo modprobe tcp_westwood
sudo modprobe tcp_veno
sudo modprobe tcp_vegas
sudo modprobe tcp_yeah
sudo modprobe tcp_cdg
sudo modprobe tcp_bic
sudo modprobe tcp_htcp
sudo modprobe tcp_hybla
sudo modprobe tcp_highspeed
sudo modprobe tcp_illinois
sudo modprobe tcp_bbr1


# sudo modprobe tcp_pcc.ko