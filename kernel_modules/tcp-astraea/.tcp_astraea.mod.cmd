savedcmd_tcp_astraea.mod := printf '%s\n'   tcp_astraea.o | awk '!x[$$0]++ { print("./"$$0) }' > tcp_astraea.mod
