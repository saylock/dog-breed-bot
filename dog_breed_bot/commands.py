import typer

app = typer.Typer(help="Dog breed bot project CLI (train/export/infer).")


@app.command()
def hello() -> None:
    """Sanity check command."""
    typer.echo("Hello! CLI is working.")


@app.command("prepare-data")
def prepare_data() -> None:
    """Prepare dataset splits + label mapping."""
    from dog_breed_bot.data.prepare_splits import main as prepare_main

    prepare_main()


@app.command()
def train() -> None:
    """Train the model."""
    from dog_breed_bot.train.train import main as train_main

    train_main()


@app.command("export-onnx")
def export_onnx() -> None:
    """Export the trained model to ONNX (placeholder for now)."""
    typer.echo("export-onnx: not implemented yet")


@app.command()
def infer(
    image: str = typer.Option(..., "--image", "-i", help="Path to input image")
) -> None:
    """Run inference on one image (placeholder for now)."""
    typer.echo(f"infer: not implemented yet. image={image}")


@app.command()
def bot() -> None:
    """Run Telegram bot (placeholder for now)."""
    typer.echo("bot: not implemented yet")
