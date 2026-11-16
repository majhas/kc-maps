<h1 align="center">Kolmogorov-Chapman Maps: Trajectory-Free Learning for Simulating Molecular Dynamics</h1>

<p align="center">
<a href="https://arxiv.org/abs/2611.12880"><img src="https://img.shields.io/badge/arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"/></a>
</p>

KC-maps is a trajectory-free method for learning the transition kernel of an SDE. We train a normalizing flow to model the distribution of the future state given the current state and time lag τ, and use it to accelerate molecular dynamics.

## Quickstart

```bash
# Clone repo
git clone https://github.com/majhas/kc-maps.git
cd kc-maps

# Create and activate a virtual environment
uv venv .venv --python=3.13
source .venv/bin/activate

# Install in editable mode
uv pip install -e .

# Install runtime dependencies
uv pip install -r requirements.txt
```

Paths are set by environment variables:

| Variable | Default | Contents |
| --- | --- | --- |
| `KCMAPS_DATA` | `./data` | training data and reference MD |
| `KCMAPS_LOGS` | `./logs` | run outputs, one directory per run |
| `KCMAPS_ROOT` | current directory | base of the two defaults above |

## Checkpoints and data

The checkpoints for AL2, AL4 and AL6, the reference MD and the generated trajectories evaluated in the paper are in one
Hugging Face repo, [`majhas/kc-maps`](https://huggingface.co/majhas/kc-maps):
```bash
hf download majhas/kc-maps --local-dir kc-maps-hf
export KCMAPS_DATA=$PWD/kc-maps-hf/data
```

Then simulate from a trained checkpoint!
```bash
python -m kcmaps.simulate checkpoint=kc-maps-hf/checkpoints/al2
python -m kcmaps.evaluate simulation=kc-maps-hf/data/trajectories/kcmaps/AL2_tau30_n16
```

A checkpoint is a directory of three files: `config.yaml` (model, system and the position scale `x_std_nm`),
`params.msgpack` (the parameters) and `start_position.npy` (the start configuration of the chains, `(atoms, 3)` in nm).

To train, build `train.npz` and `val.npz` from the source datasets with `python -m kcmaps.training.data`; see
`data/splits/`.

## Usage

```bash
# Train
python -m kcmaps.train experiment=al2

# Simulate with a maximum lag of 30 fs and 16 chains
python -m kcmaps.simulate checkpoint=<checkpoint_dir> sampler.tau_max_fs=30 sampler.chains=16

# Score a simulation against the reference MD
python -m kcmaps.evaluate simulation=<simulation_dir>
```

`simulate` writes its chains to `$KCMAPS_LOGS/simulate/<run_name>` (`run_name` defaults to a timestamp); that directory
is the `<simulation_dir>` that `evaluate` takes.

`evaluate` prints each metric as mean ± sd and writes `$KCMAPS_LOGS/evaluate/<run_name>/equilibrium.json` (and
`kinetics.json`).

A default `simulate` run lasts up to `sampler.wall_seconds` (3600 s) or `sampler.target_ns` (1000 ns per chain),
whichever comes first. Set them lower for a short run.

Every config field can be overridden from the command line; see `configs/`.

## Citation

If you use this codebase, please cite:
```bibtex
@misc{hassan2026kcmaps,
      title={Kolmogorov-Chapman Maps: Trajectory-Free Learning for Simulating Molecular Dynamics},
      author={Majdi Hassan and Marta Skreta and Dominique Beaini and Kirill Neklyudov},
      year={2026},
      eprint={2611.12880},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2611.12880},
}
```

## License

The code in this repository is licensed under the MIT License (see [LICENSE](LICENSE)).
