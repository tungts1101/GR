PROCS=2

python LSR.py --indir small_data --init 0 --alpha 0.5 --procs $PROCS
python LSR.py --indir small_data --init 1 --alpha 0.5 --procs $PROCS
