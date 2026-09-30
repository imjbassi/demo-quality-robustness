# Regenerate the results video. Run from the repository root or from viz/:
#
#   make -C viz all        train checkpoints (if missing) + render video
#   make -C viz video      render from existing checkpoints (one command)
#   make -C viz clean      remove rendered media (checkpoints kept)

PY ?= python3
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..

all: checkpoints/manifest.json video

checkpoints/manifest.json:
	cd $(ROOT) && $(PY) viz/train_video_policies.py

video:
	cd $(ROOT) && $(PY) viz/make_video.py

clean:
	rm -f media/demo_quality_video.mp4 media/main_shot.gif

.PHONY: all video clean
