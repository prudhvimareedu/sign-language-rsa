import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from backend.rsa.public_optimizer import run_public_rsa


if __name__ == "__main__":
    run_public_rsa()