savedcmd_mutant.mod := printf '%s\n'   mutant.o | awk '!x[$$0]++ { print("./"$$0) }' > mutant.mod
