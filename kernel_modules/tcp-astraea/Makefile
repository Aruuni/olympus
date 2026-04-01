# Out-of-tree kernel module build for tcp_astraea.c
obj-m += tcp_astraea.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

MODULE := tcp_astraea

# Module signing
KEYDIR   := /its/home/keys
MOK_KEY  := $(KEYDIR)/MOK.key
MOK_CRT  := $(KEYDIR)/MOK.crt
HASHALG  := sha256
SIGNFILE := $(KDIR)/scripts/sign-file

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

sign: all
	$(SIGNFILE) $(HASHALG) $(MOK_KEY) $(MOK_CRT) $(MODULE).ko

load: sign
	sudo insmod $(MODULE).ko

unload:
	-sudo rmmod $(MODULE)

reload: clean unload load

show:
	sysctl net.ipv4.tcp_congestion_control
	sysctl net.ipv4.tcp_available_congestion_control
