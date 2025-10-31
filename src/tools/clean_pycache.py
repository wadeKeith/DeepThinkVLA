# This file removes Python bytecode caches within the repository.
# Author: Cheng Yin
# Date: 2025-09
# Copyright (c) Cheng Yin. All rights reserved.
# See LICENSE file in the project root for license information.

import os
import shutil

root_dir = "./"

for dirpath, dirnames, filenames in os.walk(root_dir):
    if "__pycache__" in dirnames:
        pycache_dir = os.path.join(dirpath, "__pycache__")

        shutil.rmtree(pycache_dir)

        print(f"Removed: {pycache_dir}")


