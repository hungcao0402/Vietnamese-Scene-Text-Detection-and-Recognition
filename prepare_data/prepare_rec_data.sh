# cd augment_rec/synthtiger &&
# synthtiger -o syndata -w 16 -c 100000 -v examples/synthtiger/template.py SynthTiger examples/synthtiger/config_horizontal.yaml  &&
# synthtiger -o evaldata -w 16 -c 10000 -v examples/synthtiger/template.py SynthTiger examples/synthtiger/config1.yaml &&
# synthtiger -o synaugment -w 16 -c 20000 -v examples/synthtiger/template.py SynthTiger examples/synthtiger/config2.yaml &&
# python3 ../../create_lmdb_dataset.py

cd augment_rec/synthtiger &&
synthtiger -o syndata -w 16 -c 5 -v examples/synthtiger/template.py SynthTiger examples/synthtiger/config_horizontal.yaml  &&
synthtiger -o evaldata -w 16 -c 5 -v examples/synthtiger/template.py SynthTiger examples/synthtiger/config1.yaml &&
synthtiger -o synaugment -w 16 -c 5 -v examples/synthtiger/template.py SynthTiger examples/synthtiger/config2.yaml &&
python3 ../../create_lmdb_dataset.py