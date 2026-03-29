Build the Python extension:

  python3 setup.py build_ext --inplace

Build the listener (adjust python-config if needed):

  cc -O2 -Wall -Wextra -o astraea_listener astraea_listener.c \
     $(python3-config --includes) $(python3-config --embed --ldflags) -lpthread

Run:

  sudo ./astraea_listener \
      --mode mininet \
      --cc-name astraea \
      --py-module astraea_service \
      --py-class AstraeaService \
      --config /path/to/astraea.json \
      --model /path/to/model
