"""Conditions, training loop, Probe Head and the experiment grid.

Owner: Experiments subteam (W4, W7).
Consumes the Model and Loss interfaces; provides the Results interface (docs/interfaces.md #3).

Commands (outputs go to runs/, which git ignores; copy finished tables to results/ by hand):

    uv run python -m topkd_imat.training.run --condition KD --supervision label-free \\
        --init pretrained
    uv run python -m topkd_imat.training.grid --inits pretrained --skip-top --epochs 30
    uv run python -m topkd_imat.training.fill_distance
    uv run tensorboard --logdir runs
"""
