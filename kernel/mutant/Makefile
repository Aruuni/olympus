KBUILD_EXTRA_SYMBOLS := ../tcp-astraea/Module.symvers

ifneq ($(KERNELRELEASE),)

obj-m := mutant.o
# mutant-objs := protocol/tcp_pcc.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

# Build flags (optional)
MY_CFLAGS += -g -DDEBUG

# Module signing (Secure Boot / MOK)
SIGN_KEY  ?= /its/home/keys/MOK.key
SIGN_CERT ?= /its/home/keys/MOK.crt
SIGN_HASH ?= sha256
SIGN_TOOL ?= $(KDIR)/scripts/sign-file

MODULE := mutant
KO     := $(PWD)/$(MODULE).ko

.PHONY: all modules debug sign install clean

all: modules sign

modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

debug:
	$(MAKE) -C $(KDIR) M=$(PWD) modules EXTRA_CFLAGS="$(MY_CFLAGS)"
	$(MAKE) sign

sign:
	@test -x "$(SIGN_TOOL)" || (echo "ERROR: sign-file not found: $(SIGN_TOOL)"; exit 1)
	@test -f "$(KO)" || (echo "ERROR: module not built: $(KO)"; exit 1)
	@test -f "$(SIGN_KEY)" || (echo "ERROR: missing key: $(SIGN_KEY)"; exit 1)
	@test -f "$(SIGN_CERT)" || (echo "ERROR: missing cert: $(SIGN_CERT)"; exit 1)
	$(SIGN_TOOL) $(SIGN_HASH) $(SIGN_KEY) $(SIGN_CERT) $(KO)

install: all
	sudo $(MAKE) -C $(KDIR) M=$(PWD) modules_install
	sudo depmod -a

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

endif