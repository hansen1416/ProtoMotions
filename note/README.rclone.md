## Check configured remotes

`rclone listremotes`

## To see details:

`rclone config show`

## To test your Google Drive remote

rclone lsd gdrive:
rclone ls gdrive: --max-depth 1

## Mount one Google Drive folder locally

rclone mount gdrive:humos_output /mnt/gdrive_humos_output --daemon

## Check the mount status

# Check the mount point exists and is a mounted filesystem
mountpoint /mnt/gdrive_humos_output

# Check it is mounted as rclone/fuse
df -hT /mnt/gdrive_humos_output

# Check from /proc/mounts
grep gdrive_humos_output /proc/mounts

# Metadata check
stat /mnt/gdrive_humos_output

# Disk usage view
rclone size gdrive:humos_output

## Unmount

fusermount -u /mnt/gdrive_humos_output

## Download google drive to local

nohup rclone copy gdrive:humos_output /media/hlz/R/humos_output \
    --progress \
    --transfers 4 \
    --checkers 8 \
    --drive-chunk-size 64M \
    --log-file rclone_download.log \
    --log-level INFO > rclone_stdout.log 2>&1 &

## Upload remote to gdrive

rclone copy hhi_single_motion_multi_shape.zip gdrive:/ckpt/

# R2

`rclone lsd r2:proto-data`

## Copy local files to R2

rclone copy /home/hlz/datasets/humos_proto/merged4/ \
    r2:proto-data/merged4/ \
    --transfers=2 \
    --s3-upload-concurrency=4 \
    --s3-chunk-size=64M \
    --retries=10 \
    --retries-sleep=30s \
    --low-level-retries=20 \
    --progress


## Download from R2

rclone copy r2:proto-data/merged4/ /workspace/datasets/merged4/ \
    --transfers=4 \
    --s3-download-concurrency=16 \
    --s3-chunk-size=128M \
    --progress