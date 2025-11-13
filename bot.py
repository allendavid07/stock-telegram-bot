import asyncio
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import os
import sqlite3

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
# Prefer environment variables for cloud deployment
BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN or TELEGRAM_BOT_TOKEN in the environment first.")


JOURNAL_FILE = "swing_journal.csv"
DB_FILE = "swing_signals.db"
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

NIFTY500_DF = None
NIFTY500_SYMBOLS = []
LAST_SCAN_RESULTS = []  # cached latest scan results

# OpenAI client (advisor mode)
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------
# DB INIT
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            symbol TEXT,
            price REAL,
            signals TEXT,
            buy REAL,
            target REAL,
            stoploss REAL,
            industry TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# -------------------------------------------------
# UTIL – Load NIFTY 500
# -------------------------------------------------
def load_nifty500():
    global NIFTY500_DF, NIFTY500_SYMBOLS
    if NIFTY500_SYMBOLS:
        return NIFTY500_SYMBOLS

    df = pd.read_csv(NIFTY500_URL)
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["SymbolNS"] = df["Symbol"] + ".NS"
    NIFTY500_DF = df
    NIFTY500_SYMBOLS = df["SymbolNS"].tolist()
    logger.info("Loaded %d NIFTY 500 symbols", len(NIFTY500_SYMBOLS))
    return NIFTY500_SYMBOLS


def get_industry(symbol_with_ns: str) -> str:
    global NIFTY500_DF
    if NIFTY500_DF is None:
        load_nifty500()
    sym = symbol_with_ns.replace(".NS", "").strip()
    match = NIFTY500_DF.loc[NIFTY500_DF["Symbol"] == sym]
    if not match.empty and "Industry" in match.columns:
        return str(match["Industry"].iloc[0])
    return "Unknown"


# -------------------------------------------------
# RSI CALC
# -------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# -------------------------------------------------
# CORE – ANALYSE SINGLE STOCK
# -------------------------------------------------
def analyze_stock(symbol):
    try:
        data = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if data is None or data.empty or len(data) < 50:
            return None

        data["SMA20"] = data["Close"].rolling(20).mean()
        data["SMA50"] = data["Close"].rolling(50).mean()
        data["RSI"] = compute_rsi(data["Close"])

        last = data.iloc[-1]
        prev = data.iloc[-2]

        price = float(last["Close"])
        signals = []

        # Bullish SMA crossover
        if last["SMA20"] > last["SMA50"] and prev["SMA20"] <= prev["SMA50"]:
            signals.append("Bullish SMA20/50 Crossover")

        # RSI oversold
        if last["RSI"] < 35:
            signals.append("RSI Oversold (<35)")

        # Volume spike
        vol_mean = data["Volume"].tail(20).mean()
        if last["Volume"] > 1.8 * vol_mean:
            signals.append("Volume Spike (>1.8x 20D avg)")

        if not signals:
            return None

        buy_price = round(price * 0.985, 2)
        target_price = round(price * 1.03, 2)
        stoploss = round(price * 0.97, 2)

        return {
            "symbol": symbol.replace(".NS", ""),
            "symbol_full": symbol,
            "price": round(price, 2),
            "signals": signals,
            "buy": buy_price,
            "target": target_price,
            "stoploss": stoploss,
        }

    except Exception as e:
        logger.warning("Error analysing %s: %s", symbol, e)
        return None


# -------------------------------------------------
# JOURNAL LOGGING (CSV + SQLite)
# -------------------------------------------------
def log_journal(entry):
    # CSV
    df = pd.DataFrame([entry])
    try:
        if os.path.exists(JOURNAL_FILE):
            old = pd.read_csv(JOURNAL_FILE)
            df = pd.concat([old, df], ignore_index=True)
    except Exception as e:
        logger.warning("Error reading journal CSV: %s", e)
    df.to_csv(JOURNAL_FILE, index=False)

    # SQLite
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO signals (ts, symbol, price, signals, buy, target, stoploss, industry)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.get("datetime"),
                entry.get("symbol"),
                entry.get("price"),
                entry.get("signals"),
                entry.get("buy"),
                entry.get("target"),
                entry.get("stoploss"),
                entry.get("industry", "Unknown"),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("Error inserting into DB: %s", e)


# -------------------------------------------------
# SCAN UNIVERSE
# -------------------------------------------------
def scan_universe():
    global LAST_SCAN_RESULTS
    symbols = load_nifty500()

    results = []
    for symbol in symbols:
        stock_data = analyze_stock(symbol)
        if not stock_data:
            continue

        industry = get_industry(symbol)
        stock_data["industry"] = industry
        results.append(stock_data)

        log_journal(
            {
                "datetime": datetime.now().isoformat(timespec="seconds"),
                "symbol": stock_data["symbol"],
                "price": stock_data["price"],
                "signals": "; ".join(stock_data["signals"]),
                "buy": stock_data["buy"],
                "target": stock_data["target"],
                "stoploss": stock_data["stoploss"],
                "industry": industry,
            }
        )

    LAST_SCAN_RESULTS = results
    return results


# -------------------------------------------------
# SECTOR HEATMAP
# -------------------------------------------------
def build_sector_heatmap(results):
    if not results:
        return "No sector data – no candidates today."

    sector_counts = {}
    for r in results:
        sector = r.get("industry", "Unknown")
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    items = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)

    lines = ["📊 *Sector-wise Heatmap*"]
    for sector, count in items[:15]:
        bar = "🟩" * min(count, 10)
        lines.append(f"{sector}: {bar} ({count})")

    return "\n".join(lines)


# -------------------------------------------------
# GTT ORDER CSV
# -------------------------------------------------
def generate_gtt_file(results, quantity=1, filename="gtt_orders.csv"):
    if not results:
        return None

    rows = []
    for r in results:
        rows.append(
            {
                "symbol": r["symbol"],
                "transaction_type": "BUY",
                "quantity": quantity,
                "limit_price": r["buy"],
                "trigger_price": round(r["buy"] * 0.995, 2),
                "target_price": r["target"],
                "stoploss": r["stoploss"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return filename


# -------------------------------------------------
# CHART GENERATION
# -------------------------------------------------
def create_price_chart(symbol: str):
    # symbol can be "BEL" or "BEL.NS"
    if not symbol.upper().endswith(".NS"):
        symbol_full = symbol.upper() + ".NS"
    else:
        symbol_full = symbol.upper()

    data = yf.download(symbol_full, period="6mo", interval="1d", progress=False)
    if data is None or data.empty:
        return None

    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()

    plt.figure(figsize=(8, 4))
    plt.plot(data.index, data["Close"], label="Close")
    plt.plot(data.index, data["SMA20"], label="SMA20")
    plt.plot(data.index, data["SMA50"], label="SMA50")
    plt.title(f"{symbol.upper()} – 6M Price with SMA20/50")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    fname = f"{symbol.upper()}_chart.png"
    plt.savefig(fname)
    plt.close()
    return fname


# -------------------------------------------------
# ADVISOR MODE – OpenAI HELPERS
# -------------------------------------------------
def ai_generate(text: str) -> str:
    """Call OpenAI and return advisor text."""
    if client is None:
        return "AI advisor is not configured. Please set OPENAI_API_KEY in the environment."

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a calm, conservative swing-trading assistant for Indian equities. "
                        "You explain things clearly using simple language, and you always include a risk/disclaimer line. "
                        "Never claim to guarantee returns. Never give personalised investment advice; "
                        "stick to educational, analytical commentary."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=600,
            temperature=0.25,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("OpenAI error: %s", e)
        return "AI advisor is temporarily unavailable. Please try again later."


def build_stock_prompt(symbol: str, analysis: dict | None):
    base = f"Stock: {symbol}\n"
    if analysis is None:
        base += (
            "No strong swing signals were detected from the technical scan, "
            "but please provide a high-level educational view on how a swing trader "
            "might think about this stock in general (without giving personalised advice).\n"
        )
        return base

    base += (
        f"Last price: ₹{analysis['price']}\n"
        f"Signals: {', '.join(analysis['signals'])}\n"
        f"Suggested buy zone (from scanner): around ₹{analysis['buy']}\n"
        f"Target (from scanner): ₹{analysis['target']}\n"
        f"Stoploss (from scanner): ₹{analysis['stoploss']}\n\n"
        "Using this information, explain:\n"
        "1) What these signals broadly indicate for swing trading.\n"
        "2) How a cautious swing trader might plan entries, exits, and position sizing.\n"
        "3) Key risks and what could go wrong.\n"
        "End with one clear disclaimer line."
    )
    return base


def build_scan_prompt(results: list[dict]):
    if not results:
        return (
            "The scanner found no swing candidates today from the NIFTY 500 universe. "
            "Explain what this might mean for a swing trader and how they should think "
            "about capital protection and patience."
        )

    # Top 10 only to keep prompt small
    subset = results[:10]
    lines = ["Swing scanner results (subset):"]
    for r in subset:
        lines.append(
            f"{r['symbol']} | Sector: {r.get('industry','Unknown')} | "
            f"Price: {r['price']} | Signals: {', '.join(r['signals'])} | "
            f"Buy: {r['buy']} | Target: {r['target']} | SL: {r['stoploss']}"
        )

    lines.append(
        "\nUsing this data, give:\n"
        "1) A short market mood summary for swing trades.\n"
        "2) Which sectors look relatively active (based only on this subset).\n"
        "3) General risk management guidance for a trader using such a scanner.\n"
        "Keep it under 250–300 words and end with a disclaimer."
    )
    return "\n".join(lines)


# -------------------------------------------------
# TELEGRAM HANDLERS – BASIC
# -------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📈 *Swing Scanner Bot v3 – Advisor Mode*\n\n"
        "Core commands:\n"
        "/scan – Scan NIFTY 500 now\n"
        "/alerts_on – 30-min realtime alerts\n"
        "/alerts_off – Stop alerts\n"
        "/daily_on – Daily morning report (9:15 IST)\n"
        "/daily_off – Stop daily report\n"
        "/gtt – Generate GTT CSV from latest scan\n"
        "/chart SYMBOL – Price chart PNG (e.g. /chart BEL)\n\n"
        "Advisor mode:\n"
        "/explain SYMBOL – AI explanation of that stock's swing setup\n"
        "/ai_scan – AI commentary on the latest scan\n"
        "/ask your question – General swing-trading question (e.g. /ask How to size positions?)\n"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Scanning NIFTY 500 for swing setups…")

    results = await asyncio.to_thread(scan_universe)

    if not results:
        await update.message.reply_text("No strong swing setups found right now.")
        return

    # limit to first 20 for message length
    lines = ["🔥 *Swing Trading Candidates*"]
    for r in results[:20]:
        lines.append(
            f"\n📌 *{r['symbol']}* ({r.get('industry','Unknown')})\n"
            f"Price: ₹{r['price']}\n"
            f"Signals: {', '.join(r['signals'])}\n"
            f"Buy: ₹{r['buy']} | Target: ₹{r['target']} | SL: ₹{r['stoploss']}"
        )

    heatmap = build_sector_heatmap(results)

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    await update.message.reply_text(heatmap, parse_mode="Markdown")


# ------------------ ALERTS (30 MIN) -----------------
async def alerts_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    job_name = f"alerts_{chat_id}"

    # remove any old job first
    current_jobs = context.job_queue.get_jobs_by_name(job_name)
    for j in current_jobs:
        j.schedule_removal()

    # 1800 seconds = 30 minutes
    context.job_queue.run_repeating(
        alerts_job,
        interval=1800,
        first=10,
        chat_id=chat_id,
        name=job_name,
    )

    await update.message.reply_text(
        "✅ 30-minute alerts enabled for this chat.\n"
        "You will receive swing candidates automatically."
    )


async def alerts_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    job_name = f"alerts_{chat_id}"
    jobs = context.job_queue.get_jobs_by_name(job_name)
    if not jobs:
        await update.message.reply_text("No active alerts for this chat.")
        return
    for j in jobs:
        j.schedule_removal()
    await update.message.reply_text("⏹ Alerts turned off for this chat.")


async def alerts_job(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    chat_id = job.chat_id

    results = await asyncio.to_thread(scan_universe)
    if not results:
        await context.bot.send_message(chat_id, text="⚠️ No swing setups this cycle.")
        return

    # send top 5
    lines = ["⏰ *30-min Alert – New Swing Candidates*"]
    for r in results[:5]:
        lines.append(
            f"\n📌 *{r['symbol']}* ({r.get('industry','Unknown')})\n"
            f"Price: ₹{r['price']} | Buy: ₹{r['buy']} | Target: ₹{r['target']} | SL: ₹{r['stoploss']}\n"
            f"Signals: {', '.join(r['signals'])}"
        )

    heatmap = build_sector_heatmap(results)

    await context.bot.send_message(chat_id, "\n".join(lines), parse_mode="Markdown")
    await context.bot.send_message(chat_id, heatmap, parse_mode="Markdown")


# ------------------ DAILY MORNING REPORT -----------------
async def daily_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    job_name = f"daily_{chat_id}"

    # remove any old jobs
    for j in context.job_queue.get_jobs_by_name(job_name):
        j.schedule_removal()

    # schedule at 9:15 AM IST
    report_time = dtime(hour=9, minute=15, tzinfo=ZoneInfo("Asia/Kolkata"))
    context.job_queue.run_daily(
        daily_job,
        time=report_time,
        chat_id=chat_id,
        name=job_name,
    )

    await update.message.reply_text(
        "🌅 Daily morning report enabled (around 9:15 AM IST)."
    )


async def daily_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    job_name = f"daily_{chat_id}"
    jobs = context.job_queue.get_jobs_by_name(job_name)
    if not jobs:
        await update.message.reply_text("No daily report configured for this chat.")
        return
    for j in jobs:
        j.schedule_removal()
    await update.message.reply_text("🛑 Daily morning report disabled.")


async def daily_job(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    chat_id = job.chat_id

    results = await asyncio.to_thread(scan_universe)
    if not results:
        await context.bot.send_message(chat_id, "🌅 Daily report: No swing setups today.")
        return

    lines = ["🌅 *Daily Swing Report*"]
    for r in results[:15]:
        lines.append(
            f"\n📌 *{r['symbol']}* ({r.get('industry','Unknown')})\n"
            f"Price: ₹{r['price']} | Buy: ₹{r['buy']} | Target: ₹{r['target']} | SL: ₹{r['stoploss']}\n"
            f"Signals: {', '.join(r['signals'])}"
        )

    heatmap = build_sector_heatmap(results)

    await context.bot.send_message(chat_id, "\n".join(lines), parse_mode="Markdown")
    await context.bot.send_message(chat_id, heatmap, parse_mode="Markdown")


# ------------------ GTT FILE COMMAND -----------------
async def gtt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_SCAN_RESULTS
    if not LAST_SCAN_RESULTS:
        await update.message.reply_text(
            "No cached scan results. Run /scan first to generate candidates."
        )
        return

    filename = generate_gtt_file(LAST_SCAN_RESULTS)
    if not filename or not os.path.exists(filename):
        await update.message.reply_text("Failed to generate GTT file.")
        return

    with open(filename, "rb") as f:
        await update.message.reply_document(
            document=f,
            filename=filename,
            caption="📄 GTT order template generated from latest scan.",
        )


# ------------------ CHART COMMAND -----------------
async def chart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /chart SYMBOL  (e.g. /chart BEL)")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"Generating chart for {symbol}…")

    fname = await asyncio.to_thread(create_price_chart, symbol)
    if not fname or not os.path.exists(fname):
        await update.message.reply_text("Could not generate chart. Try another symbol.")
        return

    with open(fname, "rb") as f:
        await update.message.reply_photo(
            photo=f,
            caption=f"{symbol} – 6M price with SMA20/50",
        )


# ------------------ ADVISOR COMMANDS -----------------
async def explain_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /explain SYMBOL  (e.g. /explain BEL)")
        return

    symbol = context.args[0].upper()
    symbol_full = symbol if symbol.endswith(".NS") else symbol + ".NS"

    await update.message.reply_text(f"Running scan and AI explanation for {symbol}…")

    analysis = await asyncio.to_thread(analyze_stock, symbol_full)
    prompt = build_stock_prompt(symbol, analysis)
    answer = await asyncio.to_thread(ai_generate, prompt)

    await update.message.reply_text(answer)


async def ai_scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_SCAN_RESULTS
    await update.message.reply_text("Asking AI to interpret the latest scan…")

    # Use cached results if available, else run a fresh scan
    results = LAST_SCAN_RESULTS
    if not results:
        results = await asyncio.to_thread(scan_universe)

    prompt = build_scan_prompt(results)
    answer = await asyncio.to_thread(ai_generate, prompt)
    await update.message.reply_text(answer)


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Capture the whole text after "/ask "
    text = update.message.text or ""
    parts = text.split(" ", 1)
    if len(parts) < 2 or not parts[1].strip():
        await update.message.reply_text("Usage: /ask your question here")
        return

    question = parts[1].strip()
    await update.message.reply_text("Thinking about that…")

    prompt = (
        "Answer this swing-trading / stock-market question for an Indian retail trader. "
        "Be practical, conservative, and include risk notes.\n\n"
        f"Question: {question}"
    )
    answer = await asyncio.to_thread(ai_generate, prompt)
    await update.message.reply_text(answer)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
async def main():
    init_db()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.add_handler(CommandHandler("alerts_on", alerts_on))
    app.add_handler(CommandHandler("alerts_off", alerts_off))
    app.add_handler(CommandHandler("daily_on", daily_on))
    app.add_handler(CommandHandler("daily_off", daily_off))
    app.add_handler(CommandHandler("gtt", gtt_cmd))
    app.add_handler(CommandHandler("chart", chart_cmd))

    # Advisor commands
    app.add_handler(CommandHandler("explain", explain_cmd))
    app.add_handler(CommandHandler("ai_scan", ai_scan_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))

    print("Bot is running with scanner + alerts + daily + charts + GTT + DB + advisor mode…")
    await app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
