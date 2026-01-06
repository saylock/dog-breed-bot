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
def export_onnx(
    ckpt_path: str = "artifacts/pt/best.ckpt",
    out_path: str = "artifacts/onnx/model.onnx",
) -> None:
    from dog_breed_bot.export.onnx_export import export_onnx as _export

    out = _export(ckpt_path=ckpt_path, out_path=out_path)
    print(f"✅ Exported ONNX to: {out}")


@app.command()
def infer(
    image_path: str = typer.Option(..., "--image", "-i"),
    onnx_path: str = "artifacts/onnx/model.onnx",
) -> None:
    from dog_breed_bot.infer.onnx_infer import predict

    print(predict(image_path=image_path, onnx_path=onnx_path))


@app.command()
def bot() -> None:
    """Run Telegram bot (placeholder for now)."""
    typer.echo("bot: not implemented yet")
