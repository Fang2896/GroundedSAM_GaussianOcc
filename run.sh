# cd /your_path/GroundedSAM_GaussianOcc

config=./configs/ddad_volume.txt

# DDAD
python -m torchrun --nproc_per_node 4 groundedsam_generate_sem_ddad.py --config $config
