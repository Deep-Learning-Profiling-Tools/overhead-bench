./profile_wrapper.sh nsys profile --trace=cuda --sample=none --cpuctxsw=none python training.py
./profile_wrapper.sh proton training.py
./profile_wrapper.sh python training.py --profiler torch
