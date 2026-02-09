import os
import glob
import tarfile
import zipfile
import tempfile
import urllib.request
import hashlib
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Dependency:
    name: str
    url: str
    sha256: str
    extract_path: Optional[str] = None
    include_paths: List[str] = field(default_factory=lambda: ["."])
    source_globs: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)


DEPENDENCIES = [
    Dependency(
        name="openfst",
        url="https://github.com/weimeng23/openfst/archive/refs/tags/1.8.4.zip",
        sha256="1dc6b2afd4cb4255d41d1d558d9677b9fb9451c3932ea385f9d480170a9b8e4b",
        extract_path="openfst-1.8.4",
        include_paths=["src/include"],
        source_globs=["src/lib/*.cc"]
    ),
    Dependency(
        name="boost",
        url="https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz",
        sha256="66a469b6e608a51f8347236f4912e27dc5c60c60d7d53ae9bfe4683316c6f04c",
        extract_path="boost_1_82_0",
        include_paths=["."]
    ),
    Dependency(
        name="kenlm",
        url="https://github.com/kpu/kenlm/archive/4cb443e60b7bf2c0ddf3c745378f76cb59e254e5.zip",
        sha256="eea5d0f70513ad6664fb7861bd55af888cbc967eef289a0429f52396f1808a59",
        extract_path="kenlm-4cb443e60b7bf2c0ddf3c745378f76cb59e254e5",
        include_paths=["."], # Kenlm uses includes like "util/..." so root is needed
        source_globs=[
            "util/*.cc", 
            "lm/*.cc", 
            "util/double-conversion/*.cc"
        ],
        exclude_patterns=["main.cc", "test.cc"]
    ),
    Dependency(
        name="threadpool",
        url="https://github.com/progschj/ThreadPool/archive/9a42ec1329f259a5f4881a291db1dcb8f2ad9040.zip",
        sha256="18854bb7ecc1fc9d7dda9c798a1ef0c81c2dd331d730c76c75f648189fa0c20f",
        extract_path="ThreadPool-9a42ec1329f259a5f4881a291db1dcb8f2ad9040",
        include_paths=["."]
    ),
    Dependency(
        name="utfcpp",
        url="https://github.com/nemtrif/utfcpp/archive/refs/tags/v4.0.9.zip",
        sha256="73802895d0cf7b000cdf8e6ee5d69b963a829d4ea419562afd8f190adef87d5f",
        extract_path="utfcpp-4.0.9",
        include_paths=["source"]
    )
]


class CustomBuildExtension(BuildExtension):
    """
    A custom build command that:
    1. Creates a temporary directory.
    2. Downloads and extracts C++ dependencies there.
    3. Updates the C++ extension sources/includes to point to the temp dir.
    4. Performs compiler checks using the active compiler.
    """

    def build_extensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:            
            self._prepare_dependencies(temp_dir)
            # Inject paths into the extension, assuming there is only one
            ext = self.extensions[0]
            self._update_extension_paths(ext, temp_dir)
            self._configure_compression_libs(ext)
            super().build_extensions()

    def _prepare_dependencies(self, base_dir: str) -> None:
        for dep in DEPENDENCIES:
            filename = dep.url.split('/')[-1]
            download_target = os.path.join(base_dir, filename)
            
            print(f"[{dep.name}] Downloading...")
            urllib.request.urlretrieve(dep.url, download_target)

            print(f"[{dep.name}] Verifying SHA256...")
            if not self._verify_checksum(download_target, dep.sha256):
                raise ValueError(
                    f"Checksum mismatch for {dep.name}.\n"
                    f"File: {dep.url}\n"
                    f"Expected: {dep.sha256}"
                )

            print(f"[{dep.name}] Extracting...")
            if filename.endswith('.zip'):
                with zipfile.ZipFile(download_target, 'r') as z:
                    z.extractall(base_dir)
            elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                with tarfile.open(download_target, 'r') as t:
                    t.extractall(base_dir)

    def _verify_checksum(self, file_path: str, expected_hash: str) -> bool:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in 64kb chunks to avoid memory issues with large files
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == expected_hash

    def _update_extension_paths(self, ext, base_dir):
        for dep in DEPENDENCIES:
            root_dir = os.path.join(base_dir, dep.extract_path)

            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"Expected extracted directory not found: {root_dir}. "
                                        f"Check if 'extract_path' matches the archive structure.")

            for inc_path in dep.include_paths:
                ext.include_dirs.append(os.path.join(root_dir, inc_path))

            for glob_pattern in dep.source_globs:
                # Resolve full glob path: /tmp/xyz/kenlm-master/util/*.cc
                full_pattern = os.path.join(root_dir, glob_pattern)
                found_files = glob.glob(full_pattern)
                
                if dep.exclude_patterns:
                    found_files = [
                        f for f in found_files 
                        if not any(f.endswith(exclude) for exclude in dep.exclude_patterns)
                    ]
                
                ext.sources.extend(found_files)

    def _configure_compression_libs(self, ext):
        """
        Uses the active compiler to check if headers exist.
        """
        compiler = self.compiler
        
        def has_header(header_name):
            try:
                # Try to compile a trivial file that includes the header
                with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
                    f.write(f"#include <{header_name}>\nint main() {{ return 0; }}")
                    fname = f.name
                
                # Try to compile
                try:
                    # compiler.compile returns a list of objects on success
                    compiler.compile([fname], output_dir=os.path.dirname(fname))
                    return True
                except Exception:
                    return False
                finally:
                    # Cleanup dummy files
                    if os.path.exists(fname): os.remove(fname)
                    # Cleanup object file (compiler adds .o or .obj)
                    obj_name = fname.replace('.cpp', '.o').replace('.cpp', '.obj')
                    if os.path.exists(obj_name): os.remove(obj_name)
            except Exception:
                return False

        if has_header("zlib.h"):
            print("Found zlib.h, enabling ZLIB support.")
            ext.extra_compile_args.append('-DHAVE_ZLIB')
            ext.libraries.append('z')
        
        if has_header("bzlib.h"):
            print("Found bzlib.h, enabling BZ2 support.")
            ext.extra_compile_args.append('-DHAVE_BZLIB')
            ext.libraries.append('bz2')
            
        if has_header("lzma.h"):
            print("Found lzma.h, enabling LZMA support.")
            ext.extra_compile_args.append('-DHAVE_XZLIB')
            ext.libraries.append('lzma')


setup(
    packages=find_packages(exclude=["build", "third_party", "tests"]),
    ext_modules=[
        CppExtension(
            name='ctcdecode._ext.ctc_decode',
            package=True,
            with_cuda=False, 
            # We populate sources/includes dynamically in CustomBuildExtension
            sources=glob.glob('ctcdecode/src/*.cpp'), 
            include_dirs=include_paths(),
            extra_compile_args=['-O3', '-DKENLM_MAX_ORDER=6', '-std=c++17', '-fPIC', '-DINCLUDE_KENLM']
        )
    ],
    cmdclass={'build_ext': CustomBuildExtension}
)