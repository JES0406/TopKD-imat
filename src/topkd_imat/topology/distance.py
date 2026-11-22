"""Topology Distance between a Teacher and a Student Latent Point Cloud (Results column).

Stub until the Topology subteam provides it; Experiments fills the ``topology_distance``
column afterwards from the saved clouds, so nothing blocks on this.
"""


def topology_distance(teacher_cloud, student_cloud) -> float:
    """Distance between two clouds of shape ``(n, dim_t)`` and ``(n, dim_s)`` (numpy or torch)."""
    raise NotImplementedError(
        "topology_distance is not written yet (Topology subteam, W3). Runs record NaN in the "
        "topology_distance column until it exists."
    )
