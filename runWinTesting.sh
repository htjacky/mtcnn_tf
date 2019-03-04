#!/bin/bash
set -e
### All of your tmp data will be saved in ./tmp folder

echo "Hello! I will prepare training data and starting to training step by step."

# 1. checking dataset if OK
python testing/generate_wider_result_for_win.py --epoch=2
python testing/generate_wider_result_for_win.py --epoch=6
python testing/generate_wider_result_for_win.py --epoch=12
python testing/generate_wider_result_for_win.py --epoch=20
python testing/generate_wider_result_for_win.py --epoch=28
# 5. Done
echo "Congratulation! All stages had been done. Now you can going to testing and hope you enjoy your result."
echo "haha...bye bye"

