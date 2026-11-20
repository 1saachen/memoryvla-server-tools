import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'setup_vulkan.py'


class VulkanSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('setup_vulkan', MODULE)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_only_verified_driver_and_os_combination_is_used(self):
        url, checksum = self.mod.package_info('ubuntu', '22.04', '570.148.08', 'x86_64')
        self.assertIn('ubuntu2204/x86_64/libnvidia-gl-570_570.148.08-1ubuntu1_amd64.deb', url)
        self.assertEqual(len(checksum), 64)
        self.assertNotEqual(checksum, self.mod.package_info('ubuntu', '24.04', '570.148.08', 'x86_64')[1])
        for os_id, version, driver, arch in [('debian', '12', '570.148.08', 'x86_64'),
                                              ('ubuntu', '22.04', '570.211.01', 'x86_64'),
                                              ('ubuntu', '22.04', '570.148.08', 'aarch64')]:
            with self.assertRaises(ValueError):
                self.mod.package_info(os_id, version, driver, arch)

    def test_checksum_rejects_wrong_download(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'package.deb'
            path.write_bytes(b'invalid package')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                self.mod.verify_checksum(path, '0' * 64)

    def test_local_icd_and_environment_keep_system_driver_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d)
            lib = prefix / 'extracted/usr/lib/x86_64-linux-gnu'
            lib.mkdir(parents=True)
            (lib / 'libGLX_nvidia.so.570.148.08').touch()
            vendor = prefix / 'extracted/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
            vendor.parent.mkdir(parents=True)
            vendor.write_text('{}')
            script, env = self.mod.prepare_environment(prefix, {'LD_LIBRARY_PATH': '/existing/lib', 'KEEP': 'yes'})
            icd = json.loads(Path(env['VK_ICD_FILENAMES']).read_text())
            self.assertEqual(icd['ICD']['library_path'], str(lib / 'libGLX_nvidia.so.570.148.08'))
            self.assertTrue(env['LD_LIBRARY_PATH'].endswith(':/existing/lib'))
            self.assertEqual(env['KEEP'], 'yes')
            self.assertNotIn('VK_DRIVER_FILES', env)
            self.assertIn('unset VK_DRIVER_FILES', script.read_text())
            self.assertEqual(env['__EGL_VENDOR_LIBRARY_FILENAMES'], str(vendor))


if __name__ == '__main__':
    unittest.main()
