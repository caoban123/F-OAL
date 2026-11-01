import platform
import pytest
from src.foal.environment import get_environment_info, get_git_info


def test_git_info_structure():
    info = get_git_info()
    assert isinstance(info, dict)
    assert 'commit' in info
    assert 'branch' in info
    assert 'dirty' in info
    # In a git repo, commit should be a 40-char SHA if git is available
    if info['commit'] is not None:
        assert len(info['commit']) == 40


def test_environment_info_structure():
    env = get_environment_info()
    assert isinstance(env, dict)
    assert 'os' in env
    assert 'python' in env
    assert 'libraries' in env
    assert 'hardware' in env
    assert 'git' in env

    # OS details
    assert env['os']['system'] in ['Windows', 'Linux', 'Darwin']

    # Library checks
    libs = env['libraries']
    assert 'torch' in libs
    assert 'torchvision' in libs
    assert 'timm' in libs
    assert libs['torch'] is not None

    # Hardware details
    assert 'cuda_available' in env['hardware']
    assert isinstance(env['hardware']['cuda_available'], bool)
