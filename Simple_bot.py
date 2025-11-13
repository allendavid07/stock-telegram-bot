import logging
import asyncio
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# 🔑 Put your real token here
TELEGRAM_BOT_TOKEN = "8436471379:AAGlLjUb9sqshkmIgXa4S5UqJDSe763ejEk"

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Please set TELEGRAM_BOT_TOKEN in the code.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot is alive and responding! 🎉")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ""
    await update.message.reply_text(f"You said: {text}")


def main():
    # 🔧 Manually create and set an event loop (Python 3.14 workaround)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("echo", echo))

    logging.info("Simple bot starting…")
    app.run_polling()  # this will now find the loop we just created


if __name__ == "__main__":
    main()
