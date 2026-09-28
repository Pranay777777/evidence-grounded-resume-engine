"""Entry point: python -m grounded — serves the API and the admin UI."""

from __future__ import annotations

import logging

import uvicorn

from grounded.api.app import create_app
from grounded.config import get_settings
from grounded.logging import configure_logging


def main() -> None:
    """Start the server on the configured host and port."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logging.getLogger(__name__).info("started in %s", settings.app_env)
    uvicorn.run(create_app(), host=settings.host, port=settings.port, log_config=None)


if __name__ == "__main__":
    main()
