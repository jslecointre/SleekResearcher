import logging.config
from configparser import ConfigParser
from pathlib import Path

from deep_research.utils import get_current_dir

APP_ROOT = get_current_dir()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

config = ConfigParser()
config.read(f"{APP_ROOT}/config/app_config.ini")
logging.config.fileConfig(
    f"{APP_ROOT}/config/logging.conf",
    defaults={"logfilename": f'{APP_ROOT}/{config.get("Paths", "log_file")}'},
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
