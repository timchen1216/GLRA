# gt_folder_path=/home/caig/data/MOT17/train
# val_map_path=/home/caig/data/MOT17/17train_seqmap.txt
# track_results_path=/home/caig/repo/SparseTrack/yolox_mix17/yolox_mix17_det/track_results

gt_folder_path=/home/caig/data/MOT20/train
val_map_path=/home/caig/data/MOT20/20train_seqmap.txt
track_results_path=/home/caig/repo/SparseTrack/yolox_mix20/yolox_mix20_det/track_results

# need to change 'gt_val_half.txt' or 'gt.txt'
val_type='{gt_folder}/{seq}/gt/gt.txt'
# val_type='{gt_folder}/{seq}/gt/gt_val_half.txt'

# command
python TrackEval/scripts/run_mot_challenge.py  \
        --SPLIT_TO_EVAL train  --METRICS HOTA CLEAR Identity  --GT_FOLDER ${gt_folder_path}   \
        --SEQMAP_FILE ${val_map_path}  --SKIP_SPLIT_FOL True   --TRACKERS_TO_EVAL '' \
        --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True   --NUM_PARALLEL_CORES 8   --PLOT_CURVES False   \
        --TRACKERS_FOLDER  ${track_results_path}  \
        --GT_LOC_FORMA ${val_type}