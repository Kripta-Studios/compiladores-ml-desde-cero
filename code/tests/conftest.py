"""Run the numerical tests on an explicitly selected device backend."""
from functools import partial
import pytest
from lumbre import Program


def pytest_addoption(parser):
    parser.addoption('--backend', choices=('cpu', 'cuda', 'hip'), default='cpu',
                     help='Backend for Lumbre Programs; requested devices must work.')


@pytest.fixture(autouse=True)
def select_backend(request, monkeypatch):
    backend = request.config.getoption('--backend')
    if backend != 'cpu' and hasattr(request.module, 'Program'):
        monkeypatch.setattr(request.module, 'Program', partial(Program, backend=backend))
