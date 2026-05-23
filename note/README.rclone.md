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

