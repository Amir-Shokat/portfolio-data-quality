import pytest

from portfolio_dq.generator import make_dataset


@pytest.fixture(scope="session")
def clean():
    return make_dataset(seed=7, defects="none")


@pytest.fixture(scope="session")
def broken():
    return make_dataset(seed=7, defects="all")
