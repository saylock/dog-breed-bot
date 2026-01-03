import typer

app = typer.Typer(help="Dog breed bot project CLI (train/export/infer).")


@app.command()
def hello() -> None:
    """Sanity check command."""
    typer.echo("Hello! CLI is working.")


@app.command("prepare-data")
def prepare_data() -> None:
    """Prepare dataset splits + label mapping (placeholder for now)."""
    typer.echo("prepare-data: not implemented yet")


@app.command()
def train() -> None:
    """Train the model (placeholder for now)."""
    typer.echo("train: not implemented yet")


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
