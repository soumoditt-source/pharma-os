"""Entry point for ``python -m server``."""

from server.app import main


if __name__ == "__main__":
    main(host="127.0.0.1", port=8000, workers=1)
