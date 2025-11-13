import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import yfinance as yf
import feedparser
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# ============================================================
#  CONFIG – EDIT THESE
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "PASTE_YOUR_OPENAI_API_KEY_HERE")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "PASTE_YOUR_TELEGRAM_BOT_TOKEN_HERE")

if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("PASTE_"):
    raise RuntimeError("Set OPENAI_API_KEY in code or environment first.")
if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN.startswith("PASTE_"):
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN in code or environment first.")

# ============================================================
#  LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ============================================================
#  OPENAI CLIENT
# ============================================================

client = OpenAI(api_key=OPENAI_API_KEY)


async def ask_gpt(system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI chat completion with basic safety around errors."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=900,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.exception("Error calling OpenAI: %s", e)
        return (
            "⚠️ I ran into an error while talking to the AI backend. "
            "Please try again after some time."
        )


# ============================================================
#  STOCK DATA – yfinance
# ============================================================

def get_stock_snapshot(symbol: str) -> dict:
    """
    Fetch basic stock info & 6-month history from yfinance.
    Used for /fundamental and /technical.
    """
    ticker = yf.Ticker(symbol)

    hist = ticker.history(period="6mo")
    if hist.empty:
        raise ValueError("No price history found for this symbol.")

    last_close = float(hist["Close"].iloc[-1])
    ma20 = float(hist["Close"].tail(20).mean())
    ma50 = float(hist["Close"].tail(50).mean()) if len(hist) >= 50 else None
    ma200 = float(hist["Close"].tail(200).mean()) if len(hist) >= 200 else None

    try:
        info = ticker.info
    except Exception as e:
        logger.warning("Error fetching ticker.info for %s: %s", symbol, e)
        info = {}

    snapshot = {
        "symbol": symbol,
        "last_close": last_close,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
        "dividend_yield": info.get("dividendYield"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "long_name": info.get("longName") or info.get("shortName") or symbol,
        "currency": info.get("currency"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
    }

    return snapshot


def get_current_price(symbol: str) -> Optional[float]:
    """
    Try to get the latest tradable price for symbol.
    Uses fast_info if available, falls back to intraday history.
    """
    ticker = yf.Ticker(symbol)
    # fast_info is usually the quickest
    try:
        fi = ticker.fast_info
        for key in ("lastPrice", "last_price", "regularMarketPrice", "currentPrice"):
            if key in fi and fi[key] is not None:
                return float(fi[key])
    except Exception as e:
        logger.warning("fast_info failed for %s: %s", symbol, e)

    # Fallback: use last intraday close
    try:
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning("history fallback failed for %s: %s", symbol, e)

    return None


# ============================================================
#  NEWS – Google News RSS via feedparser
# ============================================================

def get_news_for_symbol(symbol: str, max_items: int = 6):
    """
    Very simple news fetch using Google News RSS.
    """
    query = f"{symbol} stock"
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries[:max_items]:
        published = getattr(entry, "published", "")
        title = getattr(entry, "title", "No title")
        link = getattr(entry, "link", "")
        items.append(
            {
                "title": title,
                "published": published,
                "link": link,
            }
        )
    return items


# ============================================================
#  ALERTS – In-memory simple engine
# ============================================================

@dataclass
class PriceAlert:
    symbol: str
    target: float
    direction: str  # "above" or "below"


# chat_id -> list of alerts
ALERTS: Dict[int, List[PriceAlert]] = {}


def add_price_alert(chat_id: int, symbol: str, target: float) -> str:
    """
    Create a price alert. Direction decided from current price:
    - if current < target => alert when price goes ABOVE target
    - if current > target => alert when price goes BELOW target
    - if equal => trigger immediately & don't store
    """
    cur = get_current_price(symbol)
    if cur is None:
        return (
            f"⚠️ Could not fetch current price for {symbol}. "
            "Alert not created."
        )

    if cur < target:
        direction = "above"
        direction_text = "when price moves UP to or above"
    elif cur > target:
        direction = "below"
        direction_text = "when price moves DOWN to or below"
    else:
        # already at target
        return (
            f"⚠️ {symbol} is already around {target:.2f}. "
            "No alert created."
        )

    alerts = ALERTS.setdefault(chat_id, [])
    alerts.append(PriceAlert(symbol=symbol, target=target, direction=direction))

    return (
        f"✅ Alert set for {symbol} {direction_text} {target:.2f}.\n"
        f"(Current price: {cur:.2f})"
    )


def list_alerts(chat_id: int) -> str:
    alerts = ALERTS.get(chat_id, [])
    if not alerts:
        return "You have no active alerts."

    lines = ["🔔 Your active alerts:\n"]
    for idx, al in enumerate(alerts, start=1):
        arrow = "↑ above" if al.direction == "above" else "↓ below"
        lines.append(f"{idx}. {al.symbol} – {arrow} {al.target:.2f}")
    return "\n".join(lines)


def remove_alerts_for_symbol(chat_id: int, symbol: str) -> str:
    alerts = ALERTS.get(chat_id, [])
    if not alerts:
        return "You have no active alerts."

    before = len(alerts)
    alerts = [al for al in alerts if al.symbol.upper() != symbol.upper()]
    after = len(alerts)

    if after == before:
        return f"No alerts found for {symbol}."
    ALERTS[chat_id] = alerts
    return f"✅ Removed {before - after} alert(s) for {symbol}."


async def check_alerts_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Periodic job: check all alerts for all chats.
    If any alert condition is met, send a message and remove that alert.
    """
    if not ALERTS:
        return

    # Collect all symbols we need prices for
    symbols: Dict[str, float] = {}
    for alerts in ALERTS.values():
        for al in alerts:
            symbols[al.symbol] = 0.0  # placeholder

    # Fetch prices one-by-one (could be optimized to batch later)
    prices: Dict[str, Optional[float]] = {}
    for sym in symbols.keys():
        prices[sym] = get_current_price(sym)

    # Evaluate alerts
    triggered: Dict[int, List[PriceAlert]] = {}

    for chat_id, alerts in ALERTS.items():
        for al in alerts:
            price = prices.get(al.symbol)
            if price is None:
                continue
            if al.direction == "above" and price >= al.target:
                triggered.setdefault(chat_id, []).append(al)
            elif al.direction == "below" and price <= al.target:
                triggered.setdefault(chat_id, []).append(al)

    # Notify and remove triggered alerts
    for chat_id, trig_list in triggered.items():
        # Build a nice message per chat
        lines = ["🔔 Price alert(s) triggered:\n"]
        for al in trig_list:
            price = prices.get(al.symbol)
            arrow = "↑" if al.direction == "above" else "↓"
            lines.append(
                f"• {al.symbol}: {arrow} target {al.target:.2f}, "
                f"current {price:.2f if price is not None else 'N/A'}"
            )

        try:
            await context.bot.send_message(chat_id=chat_id, text="\n".join(lines))
        except Exception as e:
            logger.exception("Error sending alert message to %s: %s", chat_id, e)

        # Remove those alerts from the store
        remaining = []
        for al in ALERTS.get(chat_id, []):
            if al not in trig_list:
                remaining.append(al)
        ALERTS[chat_id] = remaining


# ============================================================
#  TELEGRAM COMMAND HANDLERS
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Hi! I’m your Stock Analysis & Alerts Bot.\n\n"
        "Commands:\n"
        "• /fundamental SYMBOL – basic fundamental-style analysis\n"
        "• /technical SYMBOL – simple technical view\n"
        "• /news SYMBOL – latest news headlines\n"
        "• /sentiment SYMBOL – sentiment based on recent news\n"
        "• /alert SYMBOL PRICE – set a price alert\n"
        "• /alerts – list your active alerts\n"
        "• /removealert SYMBOL – remove all alerts for a symbol\n\n"
        "Examples:\n"
        "  /fundamental RELIANCE.NS\n"
        "  /technical TCS.NS\n"
        "  /news HDFCBANK.NS\n"
        "  /sentiment INFY.NS\n"
        "  /alert RELIANCE.NS 2700\n\n"
        "Disclaimer: Information & education only, not investment advice. 📜"
    )
    await update.message.reply_text(text)


async def fundamental_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text(
                "Usage: /fundamental SYMBOL\nExample: /fundamental RELIANCE.NS"
            )
            return

        symbol = context.args[0].upper()
        await update.message.reply_text(f"🔍 Fetching data for {symbol}...")

        snapshot = get_stock_snapshot(symbol)
        data_text = "\n".join(f"{k}: {v}" for k, v in snapshot.items())

        system_prompt = (
            "You are a stock market analysis assistant. "
            "Explain things in simple, clear language. "
            "Use only the data provided. "
            "If some data points are missing, say they appear missing. "
            "Focus on an educational, high-level fundamental view of the stock. "
            "Always add a disclaimer that this is not investment advice."
        )

        user_prompt = f"""
Perform a basic fundamental-style analysis of this stock:

Raw snapshot data:
{data_text}

Please cover, in simple terms:
1. What kind of business this seems to be (based on name and sector/industry if available).
2. Valuation hints using P/E, P/B, dividend yield if present.
3. Price vs 52-week high/low.
4. Any hints from moving averages (last_close vs ma20/ma50/ma200).
5. A simple Bullish case (why someone might like it).
6. A simple Bearish case (why someone might be cautious).

Keep it under ~300–400 words.
"""

        answer = await ask_gpt(system_prompt, user_prompt)
        await update.message.reply_text(answer)

    except Exception as e:
        logger.exception("Error in /fundamental handler: %s", e)
        await update.message.reply_text("⚠️ Something went wrong in /fundamental.")


async def technical_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text(
                "Usage: /technical SYMBOL\nExample: /technical TCS.NS"
            )
            return

        symbol = context.args[0].upper()
        await update.message.reply_text(
            f"📊 Doing a simple technical view for {symbol}..."
        )

        snapshot = get_stock_snapshot(symbol)
        data_text = "\n".join(f"{k}: {v}" for k, v in snapshot.items())

        system_prompt = (
            "You are a technical analysis helper. "
            "You are given basic price levels and moving averages. "
            "Explain the trend in simple language. "
            "Do NOT claim to predict the future. "
            "This is not investment advice."
        )

        user_prompt = f"""
Here is the technical snapshot for {snapshot['symbol']}:

{data_text}

Using only this information, answer:
1. Is the price closer to 52-week high or low?
2. Is the price above or below its 20, 50, 200 day moving averages (when available)?
3. What does that roughly suggest about short-term vs long-term trend?
4. Provide 3–5 bullet points summarizing the technical picture.

Use simple language, no complicated jargon.
"""

        answer = await ask_gpt(system_prompt, user_prompt)
        await update.message.reply_text(answer)

    except Exception as e:
        logger.exception("Error in /technical handler: %s", e)
        await update.message.reply_text("⚠️ Something went wrong in /technical.")


async def news_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text(
                "Usage: /news SYMBOL\nExample: /news RELIANCE.NS"
            )
            return

        symbol = context.args[0].upper()
        items = get_news_for_symbol(symbol)

        if not items:
            await update.message.reply_text(f"No recent news found for {symbol}.")
            return

        lines = [f"📰 Latest news for *{symbol}*:\n"]
        for item in items:
            published = item["published"] or ""
            lines.append(
                f"• {item['title']}\n  {published}\n  {item['link']}\n"
            )

        await update.message.reply_markdown("\n".join(lines))

    except Exception as e:
        logger.exception("Error in /news handler: %s", e)
        await update.message.reply_text("⚠️ Something went wrong in /news.")


async def sentiment_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text(
                "Usage: /sentiment SYMBOL\nExample: /sentiment INFY.NS"
            )
            return

        symbol = context.args[0].upper()
        items = get_news_for_symbol(symbol, max_items=8)

        if not items:
            await update.message.reply_text(f"No recent news found for {symbol}.")
            return

        news_text_parts = []
        for i, item in enumerate(items, start=1):
            news_text_parts.append(
                f"{i}. {item['title']} ({item['published']})\n{item['link']}"
            )
        news_text = "\n\n".join(news_text_parts)

        system_prompt = (
            "You are a sentiment analysis assistant for stocks. "
            "Given a list of recent headlines, classify overall sentiment "
            "as Positive, Negative, or Neutral for the stock. "
            "Base it ONLY on the headlines and dates. "
            "Explain your reasoning in simple terms and keep it concise. "
            "Add a reminder that news sentiment can change quickly and "
            "this is not investment advice."
        )

        user_prompt = f"""
Recent news for stock {symbol}:

{news_text}

Tasks:
1. Classify the overall sentiment as Positive, Negative, or Neutral.
2. Give 3–6 bullet points explaining why.
3. Mention if the impact feels more short-term or long-term.
"""

        answer = await ask_gpt(system_prompt, user_prompt)
        await update.message.reply_text(answer)

    except Exception as e:
        logger.exception("Error in /sentiment handler: %s", e)
        await update.message.reply_text("⚠️ Something went wrong in /sentiment.")


# ---------------- ALERT COMMANDS ----------------

async def alert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /alert SYMBOL PRICE
    Example: /alert RELIANCE.NS 2700
    """
    try:
        if len(context.args) != 2:
            await update.message.reply_text(
                "Usage: /alert SYMBOL PRICE\nExample: /alert RELIANCE.NS 2700"
            )
            return

        symbol = context.args[0].upper()
        try:
            target = float(context.args[1])
        except ValueError:
            await update.message.reply_text("PRICE must be a number.")
            return

        chat_id = update.effective_chat.id
        resp = add_price_alert(chat_id, symbol, target)
        await update.message.reply_text(resp)

    except Exception as e:
        logger.exception("Error in /alert handler: %s", e)
        await update.message.reply_text("⚠️ Something went wrong in /alert.")


async def alerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /alerts – list active alerts for this chat
    """
    try:
        chat_id = update.effective_chat.id
        msg = list_alerts(chat_id)
        await update.message.reply_text(msg)
    except Exception as e:
        logger.exception("Error in /alerts handler: %s", e)
        await update.message.reply_text("⚠️ Something went wrong in /alerts.")


async def removealert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /removealert SYMBOL – remove all alerts for given symbol
    """
    try:
        if not context.args:
            await update.message.reply_text(
                "Usage: /removealert SYMBOL\nExample: /removealert RELIANCE.NS"
            )
            return

        symbol = context.args[0].upper()
        chat_id = update.effective_chat.id
        msg = remove_alerts_for_symbol(chat_id, symbol)
        await update.message.reply_text(msg)
    except Exception as e:
        logger.exception("Error in /removealert handler: %s", e)
        await update.message.reply_text("⚠️ Something went wrong in /removealert.")


# ============================================================
#  MAIN – EVENT LOOP & JOBS
# ============================================================

def main():
    # If you're on 3.14, explicitly create and set an event loop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("fundamental", fundamental_cmd))
    app.add_handler(CommandHandler("technical", technical_cmd))
    app.add_handler(CommandHandler("news", news_cmd))
    app.add_handler(CommandHandler("sentiment", sentiment_cmd))
    app.add_handler(CommandHandler("alert", alert_cmd))
    app.add_handler(CommandHandler("alerts", alerts_cmd))
    app.add_handler(CommandHandler("removealert", removealert_cmd))

    # Periodic job: check alerts every 60 seconds
    job_queue = app.job_queue
    job_queue.run_repeating(check_alerts_job, interval=60, first=15)

    logger.info("Bot starting with alerts enabled…")
    app.run_polling()


if __name__ == "__main__":
    main()
