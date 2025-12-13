"""
Entrypoint wrapper for the Deep Lagrangian Networks repo.
The repository is cloned to /workspace/deep_lagrangian_networks and installed
editable. Extend this script to call the training/eval routines you need and to
save checkpoints into /workspace/shared/models/delan.
"""

from pathlib import Path


def main() -> None:
    repo = Path("/workspace/deep_lagrangian_networks")
    print(f"DeLaN container ready. Repo at: {repo}")
    if not repo.exists():
        print("ERROR: Repository not found inside container.")
    else:
        print("TODO: implement training/eval invocation here.")


if __name__ == "__main__":
    main()
