import os
import tempfile
import unittest
from unittest import mock

import yaml

from olympus.common import runtime_config


class RuntimeConfigTest(unittest.TestCase):
    def _write_cfg(self, directory, name, cfg):
        path = os.path.join(directory, name)
        with open(path, 'w') as f:
            yaml.safe_dump(cfg, f)
        return path

    def test_agent_value_reads_resolved_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_cfg(directory, 'config.yaml', {
                'agent': {'hidden': 321},
            })
            with mock.patch.dict(os.environ, {'SAO_CONFIG': path}, clear=False):
                cfg = runtime_config.load_config()
        self.assertEqual(runtime_config.agent_value(cfg, 'hidden'), 321)

    def test_env_fallback_when_yaml_key_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_cfg(directory, 'config.yaml', {'agent': {}})
            with mock.patch.dict(
                    os.environ,
                    {'SAO_CONFIG': path, 'SAO_HIDDEN': '123'},
                    clear=False):
                cfg = runtime_config.load_config()
                value = runtime_config.agent_value(
                    cfg, 'hidden', env='SAO_HIDDEN', default=64)
        self.assertEqual(value, '123')

    def test_cache_keys_on_actual_config_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path_a = self._write_cfg(directory, 'a.yaml', {
                'agent': {'hidden': 1},
            })
            path_b = self._write_cfg(directory, 'b.yaml', {
                'agent': {'hidden': 2},
            })
            with mock.patch.dict(os.environ, {'SAO_CONFIG': path_a}, clear=False):
                cfg_a = runtime_config.load_config()
            with mock.patch.dict(os.environ, {'SAO_CONFIG': path_b}, clear=False):
                cfg_b = runtime_config.load_config()
        self.assertEqual(runtime_config.agent_value(cfg_a, 'hidden'), 1)
        self.assertEqual(runtime_config.agent_value(cfg_b, 'hidden'), 2)


if __name__ == '__main__':
    unittest.main()
