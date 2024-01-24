"""titiler.xarray"""

__version__ = "0.1.0"

import logging
import os

os.environ["ZARR_V3_EXPERIMENTAL_API"] = "1"

# the httpx library (an Arraylake dependency) somehow comes through with debug logging
logging.getLogger("httpx").setLevel(logging.WARNING)