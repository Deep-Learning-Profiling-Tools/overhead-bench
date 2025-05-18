
#!/usr/bin/env bash
echo "NO PROFILER"
./profile-wrapper.sh 2 python training.py

echo "--------------------------------------------"

echo "NSYS"
./profile-wrapper.sh 2 nsys profile --trace=cuda --sample=none --cpuctxsw=none python training.py

echo "--------------------------------------------"

echo "PROTON"
./profile-wrapper.sh 2 proton training.py

echo "--------------------------------------------"

echo "TORCH"
./profile-wrapper.sh 2 python training.py --profile_torch
