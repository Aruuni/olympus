savedcmd_mutant.ko := ld -r -m elf_x86_64 -z noexecstack --build-id=sha1  -T /its/home/mm2350/Desktop/bbr/scripts/module.lds -o mutant.ko mutant.o mutant.mod.o .module-common.o
