# Regenerate the results videos. Run from the repository root or from viz/:
#
#   make -C viz all        train checkpoints (if missing) + render both videos
#   make -C viz video      one command: re-render both videos from checkpoints
#   make -C viz clean      remove rendered media (checkpoints kept)

PY ?= python3
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..

all: checkpoints/manifest.json video

checkpoints/manifest.json:
	cd $(ROOT) && $(PY) viz/train_video_policies.py

video:
	cd $(ROOT) && $(PY) viz/make_video.py
	cd $(ROOT) && $(PY) viz/make_video2.py

clean:
	rm -f media/demo_quality_video.mp4 media/main_shot.gif \
	      media/compounding_error_video.mp4 media/matched_mse.gif

.PHONY: all video clean
