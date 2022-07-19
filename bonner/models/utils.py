from pathlib import Path
import os

_MODELS_HOME = Path(os.getenv("MODELS_HOME", str(Path.home())))
