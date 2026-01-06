from __future__ import annotations

import os
import tempfile

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from dog_breed_bot.infer.onnx_infer import predict

load_dotenv()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text("Send me a dog photo and I’ll tell the breed")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Just send a photo of a dog. I will reply with the breed name."
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or not update.message.photo:
        return

    photo = update.message.photo[-1]  # highest resolution
    tg_file = await photo.get_file()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    await tg_file.download_to_drive(tmp_path)

    try:
        breed = predict(tmp_path)
        await update.message.reply_text(f"I think it is: {breed}")
    except Exception as e:
        await update.message.reply_text(f"Sorry, I failed to run inference: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def main() -> None:
    token = os.getenv("TG_BOT_TOKEN")
    if not token:
        raise RuntimeError("TG_BOT_TOKEN is not set. Put it into .env")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
