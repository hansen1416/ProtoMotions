#!/usr/bin/env bash
# Streams the whole r2:proto-data bucket (~1.2 TiB, ~23k files) to Google Drive. Nothing is
# staged on local disk: rclone pipes each file from R2 straight into a Drive upload, so the
# pod only needs ~1-2 GB of disk (config + logs) and well under 1 GB of RAM.
#
# Prerequisites on the pod:
#   - rclone installed (curl https://rclone.org/install.sh | bash)
#   - an rclone.conf defining BOTH remotes, `r2` (S3 / Cloudflare) and `gdrive` (Drive OAuth).
#     Copy it from the PC (`rclone config file` prints the path on each machine). It holds
#     secrets, so move it over your own channel and never paste it into a chat or commit it.
#
# Usage (run from the repo root, under tmux or nohup -- the copy takes ~2 days because Drive
# caps uploads at ~750 GB per day):
#   bash tools/migrate_r2_to_gdrive.sh dry-run   # lists what would be copied, writes nothing
#   bash tools/migrate_r2_to_gdrive.sh copy      # resumable; re-running skips finished files
#   bash tools/migrate_r2_to_gdrive.sh verify    # per-folder size/count check + bucket totals
#
# Everything is overridable without editing this file, e.g.:
#   SKIP_DIRS="humos_unseen_morphology_motionlib humos_unseen_morphology_offset" \
#     bash tools/migrate_r2_to_gdrive.sh copy

set -euo pipefail

SRC="${SRC:-r2:proto-data}"
DST="${DST:-gdrive:r2-proto-data}"   # new folder: gdrive:ckpt already exists, so don't merge into the root
LOG_DIR="${LOG_DIR:-migration_logs}"
TRANSFERS="${TRANSFERS:-4}"
DRIVE_CHUNK="${DRIVE_CHUNK:-64M}"
TPS="${TPS:-10}"                     # Drive API calls/sec; keeps the 21k small clip files under its rate limit
SKIP_DIRS="${SKIP_DIRS:-}"           # space-separated top-level folders to leave in R2
WAIT_ON_LIMIT_SECS="${WAIT_ON_LIMIT_SECS:-3600}"   # sleep this long when Drive's daily cap hits; 0 = stop instead
MAX_LIMIT_WAITS="${MAX_LIMIT_WAITS:-72}"

# Small folders first so problems show up early; the 1.08 TiB per-clip folder goes last.
# difficult-motions/ is empty in R2, so it is not listed.
DIRS=(
  analysis assets eval fixes hhi_stage2_splits videos
  hard_clips_discover_lineage humos ckpt
  humos_unseen_morphology humos_unseen_morphology_motionlib humos_unseen_morphology_offset
  unseen_shapes_v2
  hhi_stage2_per_clip_refined
)

MODE="${1:-}"
case "$MODE" in
  dry-run | copy | verify) ;;
  *) echo "usage: bash tools/migrate_r2_to_gdrive.sh {dry-run|copy|verify}" >&2; exit 2 ;;
esac

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${MODE}_$(date +%Y%m%d_%H%M%S).log"

is_skipped() {
  local d
  for d in $SKIP_DIRS; do
    [ "$d" = "$1" ] && return 0
  done
  return 1
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Fail fast if either remote is missing or unreadable.
preflight() {
  log "preflight: listing $SRC"
  rclone lsf "$SRC" --s3-no-check-bucket --max-depth 1 --dirs-only >/dev/null
  log "preflight: Drive quota"
  rclone about "${DST%%:*}:" | tee -a "$LOG"
}

COMMON=(
  --s3-no-check-bucket
  --transfers "$TRANSFERS" --checkers 8
  --drive-chunk-size "$DRIVE_CHUNK"
  --drive-stop-on-upload-limit
  --tpslimit "$TPS" --tpslimit-burst "$TPS"
  --retries 10 --retries-sleep 30s --low-level-retries 20
  --stats 60s --stats-one-line
  --log-file "$LOG" --log-level INFO
)
if [ "$MODE" = "dry-run" ]; then
  COMMON+=(--dry-run)
fi

# rclone exits 7 (fatal) when --drive-stop-on-upload-limit trips. Sleep and resume; the next
# run skips everything already uploaded.
run_with_limit_wait() {
  local rc waits=0
  while true; do
    set +e
    "$@"
    rc=$?
    set -e
    if [ "$rc" -eq 7 ] && [ "$WAIT_ON_LIMIT_SECS" -gt 0 ] && [ "$waits" -lt "$MAX_LIMIT_WAITS" ]; then
      waits=$((waits + 1))
      log "Drive upload limit hit (wait $waits/$MAX_LIMIT_WAITS); sleeping ${WAIT_ON_LIMIT_SECS}s"
      sleep "$WAIT_ON_LIMIT_SECS"
      continue
    fi
    return "$rc"
  done
}

do_copy() {
  local d excludes=()
  for d in "${DIRS[@]}"; do
    if is_skipped "$d"; then
      log "skip $d (SKIP_DIRS)"
      continue
    fi
    log "== $MODE $d"
    run_with_limit_wait rclone copy "$SRC/$d" "$DST/$d" "${COMMON[@]}"
  done

  # Catch-all over the whole bucket: picks up root-level files or folders created after this
  # script was written. Folders copied above are only checked, not re-uploaded.
  for d in $SKIP_DIRS; do
    excludes+=(--exclude "/$d/**")
  done
  log "== $MODE catch-all (whole bucket)"
  run_with_limit_wait rclone copy "$SRC" "$DST" "${COMMON[@]}" ${excludes[@]+"${excludes[@]}"}
  log "$MODE finished"
}

do_verify() {
  local d failed=0
  for d in "${DIRS[@]}"; do
    if is_skipped "$d"; then
      continue
    fi
    log "== check $d"
    # R2 multipart ETags are not MD5s, so Drive hashes can't be compared; size-only is the check.
    if rclone check "$SRC/$d" "$DST/$d" --s3-no-check-bucket --size-only --one-way \
      --missing-on-dst "$LOG_DIR/missing_$d.txt" --differ "$LOG_DIR/differ_$d.txt" \
      --log-file "$LOG" --log-level NOTICE; then
      log "OK   $d"
    else
      log "FAIL $d (see $LOG_DIR/missing_$d.txt and differ_$d.txt)"
      failed=1
    fi
  done
  log "== bucket totals (compare by eye; skipped folders and the empty difficult-motions/ explain any gap)"
  rclone size "$SRC" --s3-no-check-bucket 2>&1 | tee -a "$LOG"
  rclone size "$DST" 2>&1 | tee -a "$LOG"
  return "$failed"
}

preflight
case "$MODE" in
  dry-run | copy) do_copy ;;
  verify) do_verify ;;
esac
