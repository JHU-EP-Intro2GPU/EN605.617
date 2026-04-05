all: preprocess train postprocess

preprocess: # download MNIST and verify dataset
	uv run preprocess.py

train: # MPNP on MNIST image completion
	uv run train_inpainting.py

train-quick:
	uv run train_inpainting.py --epochs 20

postprocess: # generate inpainting comparison figures
	uv run postprocess.py

sync: # sync dependencies and lock file
	uv sync

clean:
	rm -rf output/

clean-all: clean
	rm -rf .venv __pycache__ data/ uv.lock

.PHONY: all train train-quick postprocess sync clean clean-all
