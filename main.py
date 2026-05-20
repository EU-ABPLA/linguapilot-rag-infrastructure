from core.settings import SettingsError, load_settings
from core.secrets import safe_error_message
from observability.logger import get_logger


def main() -> int:
    logger = get_logger("main")
    try:
        settings = load_settings("config/settings.yaml")
    except SettingsError as exc:
        logger.error(safe_error_message(exc))
        return 1
    logger.info(f"Settings loaded: llm={settings.llm.provider}, embed={settings.embedding.provider}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
