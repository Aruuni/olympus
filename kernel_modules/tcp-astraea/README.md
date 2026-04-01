# Astraea Kernel Module for TCP

> **Note:** Before you start, ensure that you have installed the Astraea kernel image and headers.

This directory provides the Astraea CC module for the Linux Kernel TCP stack. There are two modes available:

- **`normal`**: Performs regular congestion avoidance and executes CWND reduction during TCP loss recovery.
- **`bypass`**: Executes **cong\_control** and bypasses CWND reduction during TCP loss recovery, similar to TCP BBR. (In this mode, **cong\_control** bypasses the execution of **cong\_avoid**.)

These two modules are mutually exclusive, meaning only one can be loaded at a time. To check if a module is loaded and unload it if necessary, use the following commands:

```shell
lsmod | grep tcp_astraea

# If a module is loaded, unload it:
sudo rmmod tcp_astraea 
# Or unload the bypass module:
# sudo rmmod tcp_astraea_bypass
```

## Installing the Normal Module

To install the normal module, run:

```shell
make 
make install
```

## Installing the Bypass Module

To install the bypass module, run:

```shell
make bypass=y
make install bypass=y
```
> Make sure to include Astraea in `tcp_allowed_congestion_control` in the kernel configuration.
```bash
# check current allowed congestion control
sysctl net.ipv4.tcp_allowed_congestion_control
# load bbr module if you want to use it
sudo modprobe tcp_bbr
# include astraea as allowed congestion control
sudo sysctl -w net.ipv4.tcp_allowed_congestion_control="cubic reno bbr astraea"
```
