=== Python Environment Setup ===

1. Create the virtualenv (Python 3.x required):
   python3 -m venv ./venv_astraea

2. Activate and install dependencies:
   source ./venv_astraea/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

3. Build and install the tcp_sockopt C extension:
   python setup.py build_ext --inplace
   pip install -e .

   (Requires gcc and Python dev headers: sudo apt install python3-dev)

4. Deactivate when done:
   deactivate

=== Build & Run ===

gcc -O2 -Wall -o astraea_listener astraea_listener.c && sudo -E env ASTRAEA_PYTHON="$(pwd)/venv_astraea/bin/python" ./astraea_listener --mode mininet --cc-name astraea --script /its/home/mm2350/Desktop/Olympusv2/astraea_service.py --config /its/home/mm2350/Desktop/Olympusv2/astraea/astraea.json --model /its/home/mm2350/Desktop/Olympusv2/astraea/models/py --scan-ms 10
