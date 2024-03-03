import typer
import uvicorn
from src.dependencies.get_model_service.get_model_service import get_model_service
from src.settings import __version__
from src.settings.config import settings

cli = typer.Typer()


@cli.command()
def version():
    print(__version__)


@cli.command()
def train_model():
    model_service = get_model_service()
    model_service.train_model()


@cli.command()
def test_model():
    model_service = get_model_service()
    model_service.test_model()


@cli.command()
def runserver():
    config = uvicorn.Config(
        "main:application",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=True,
    )

    if settings.app_env in [
        "devl",
    ]:
        config.reload = True

    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    cli()
