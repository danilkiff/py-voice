"""Entry point: launch the Gradio web app."""

from app import build_app


def main() -> None:
    build_app().launch(server_name="0.0.0.0")


if __name__ == "__main__":  # pragma: no cover
    main()
