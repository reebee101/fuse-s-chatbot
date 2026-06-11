import os
import re
import pandas as pd
import numpy as np
import chromadb
import calendar
from chromadb.utils import embedding_functions
from openai import OpenAI

# ── 0. CACHE ─────────────────────────────────────────────────
os.environ["HF_HOME"] = "C:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "C:/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "C:/hf_cache"
os.makedirs("C:/hf_cache", exist_ok=True)

# ── 1. LOAD ──────────────────────────────────────────────────
df = pd.read_excel(r"C:\Users\R\Downloads\cleaned_data (17).xlsx")
df.columns = df.columns.str.strip()
df.dropna(subset=["order_date", "product_id", "revenue"], inplace=True)
df["order_date"] = pd.to_datetime(df["order_date"])
df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.to_period("M")
print(f"Loaded {len(df)} rows | {df['year'].min()} → {df['year'].max()}")

# ── 2. AGGREGATE CHUNKS FOR RAG ──────────────────────────────
def build_aggregate_docs(df):
    docs, ids = [], []

    # Monthly performance chunks
    monthly = df.groupby("month").agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        orders=("revenue", "count"),
        margin=("profit_margin", "mean"),
        qty=("quantity", "sum")
    ).reset_index()
    for _, r in monthly.iterrows():
        docs.append(
            f"Month {r['month']}: Revenue={r['revenue']:,.0f} EGP, "
            f"Profit={r['profit']:,.0f} EGP, Orders={r['orders']:,}, "
            f"Margin={r['margin']:.1%}, Units Sold={r['qty']:,.0f}"
        )
        ids.append(f"monthly_{r['month']}")

    # Product performance chunks
    prod = df.groupby("product_id").agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        orders=("revenue", "count"),
        margin=("profit_margin", "mean"),
        qty=("quantity", "sum"),
        avg_price=("price", "mean")
    ).reset_index()
    for _, r in prod.iterrows():
        docs.append(
            f"Product {r['product_id']}: Total Revenue={r['revenue']:,.0f} EGP, "
            f"Total Profit={r['profit']:,.0f} EGP, Orders={r['orders']:,}, "
            f"Avg Margin={r['margin']:.1%}, Units Sold={r['qty']:,.0f}, "
            f"Avg Price={r['avg_price']:,.0f} EGP"
        )
        ids.append(f"product_{r['product_id']}")

    # Yearly performance chunks
    yearly = df.groupby("year").agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        orders=("revenue", "count"),
        margin=("profit_margin", "mean"),
        qty=("quantity", "sum")
    ).reset_index()
    for _, r in yearly.iterrows():
        docs.append(
            f"Year {int(r['year'])}: Revenue={r['revenue']:,.0f} EGP, "
            f"Profit={r['profit']:,.0f} EGP, Orders={r['orders']:,}, "
            f"Avg Margin={r['margin']:.1%}, Units Sold={r['qty']:,.0f}"
        )
        ids.append(f"year_{int(r['year'])}")

    # Product × Year chunks (for trend questions per product)
    prod_year = df.groupby(["product_id", "year"]).agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        margin=("profit_margin", "mean")
    ).reset_index()
    for _, r in prod_year.iterrows():
        docs.append(
            f"Product {r['product_id']} in {int(r['year'])}: "
            f"Revenue={r['revenue']:,.0f} EGP, Profit={r['profit']:,.0f} EGP, "
            f"Margin={r['margin']:.1%}"
        )
        ids.append(f"prod_year_{r['product_id']}_{int(r['year'])}")

    return docs, ids

documents, ids = build_aggregate_docs(df)
print(f"Built {len(documents)} aggregate chunks for indexing")

# ── 3. EMBEDDINGS ────────────────────────────────────────────
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# ── 4. CHROMA DB ─────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db_v2")
collection = chroma_client.get_or_create_collection(
    name="fuse_aggregates",
    embedding_function=embedding_function
)

if collection.count() > 0:
    print(f"Already indexed ({collection.count()} chunks)\n")
else:
    print("Indexing aggregates...\n")
    BATCH_SIZE = 200
    for i in range(0, len(documents), BATCH_SIZE):
        collection.add(
            documents=documents[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE]
        )
        print(f"  → {i} to {min(i+BATCH_SIZE, len(documents))}")
    print("Done\n")

# ── 5. RICH DATA SUMMARY ─────────────────────────────────────
yearly = df.groupby("year").agg(
    revenue=("revenue", "sum"),
    profit=("profit", "sum"),
    orders=("revenue", "count"),
    margin=("profit_margin", "mean")
).reset_index().sort_values("year")

yearly["rev_growth"] = yearly["revenue"].pct_change() * 100
yearly["profit_growth"] = yearly["profit"].pct_change() * 100

yearly_str = ""
for _, r in yearly.iterrows():
    growth = f" (YoY: {r['rev_growth']:+.1f}%)" if not np.isnan(r["rev_growth"]) else " (base year)"
    yearly_str += (
        f"  {int(r['year'])}: Revenue={r['revenue']:>14,.0f} EGP | "
        f"Profit={r['profit']:>12,.0f} EGP | "
        f"Margin={r['margin']:.1%} | Orders={r['orders']:,}{growth}\n"
    )

years_range = yearly["year"].max() - yearly["year"].min()
if years_range > 0:
    first_rev = yearly.iloc[0]["revenue"]
    last_rev  = yearly.iloc[-1]["revenue"]
    cagr = ((last_rev / first_rev) ** (1 / years_range) - 1) * 100
    cagr_str = f"  Revenue CAGR ({int(yearly['year'].min())}–{int(yearly['year'].max())}): {cagr:.1f}%\n"
else:
    cagr_str = ""

monthly_trend = (
    df.groupby("month")["revenue"]
    .sum()
    .sort_index()
    .tail(12)
)
monthly_str = "\n".join(
    f"  {str(m)}: {v:,.0f} EGP" for m, v in monthly_trend.items()
)

prod_rev   = df.groupby("product_id")["revenue"].sum().nlargest(5)
prod_prof  = df.groupby("product_id")["profit"].sum().nlargest(5)
prod_marg  = df.groupby("product_id")["profit_margin"].mean().nlargest(5)
prod_units = df.groupby("product_id")["quantity"].sum().nlargest(5)
prod_worst = df.groupby("product_id")["profit_margin"].mean().nsmallest(3)

prod_rev_str   = "\n".join(f"  {p}: {v:,.0f} EGP" for p, v in prod_rev.items())
prod_prof_str  = "\n".join(f"  {p}: {v:,.0f} EGP" for p, v in prod_prof.items())
prod_marg_str  = "\n".join(f"  {p}: {v:.1%}" for p, v in prod_marg.items())
prod_units_str = "\n".join(f"  {p}: {v:,.0f} units" for p, v in prod_units.items())
prod_worst_str = "\n".join(f"  {p}: {v:.1%}" for p, v in prod_worst.items())

monthly_avg = df.groupby(df["order_date"].dt.month)["revenue"].mean()

best_month_name = calendar.month_name[monthly_avg.idxmax()]
worst_month_name = calendar.month_name[monthly_avg.idxmin()]


data_summary = f"""
════════════════════════════════════════════════
FUSE BUSINESS INTELLIGENCE REPORT
════════════════════════════════════════════════

▸ PORTFOLIO SNAPSHOT
  Total Orders  : {len(df):,}
  Total Revenue : {df['revenue'].sum():,.0f} EGP
  Total Profit  : {df['profit'].sum():,.0f} EGP
  Avg Order Size: {df['price'].mean():,.0f} EGP
  Avg Margin    : {df['profit_margin'].mean():.1%}
  Data Range    : {df['order_date'].min().strftime('%b %Y')} → {df['order_date'].max().strftime('%b %Y')}

▸ YEAR-ON-YEAR PERFORMANCE
{yearly_str}{cagr_str}
▸ LAST 12 MONTHS — MONTHLY REVENUE
{monthly_str}

▸ SEASONALITY
  Best month (avg): {best_month_name} | Worst month (avg): {worst_month_name}

▸ TOP 5 PRODUCTS — REVENUE
{prod_rev_str}

▸ TOP 5 PRODUCTS — PROFIT
{prod_prof_str}

▸ TOP 5 PRODUCTS — MARGIN
{prod_marg_str}

▸ TOP 5 PRODUCTS — UNITS SOLD
{prod_units_str}

▸ LOWEST MARGIN PRODUCTS (watch list)
{prod_worst_str}
════════════════════════════════════════════════
"""

print(data_summary)

def generate_business_diagnostics(df):

    diagnostics = []

    yearly = (
        df.groupby("year")
        .agg(
            revenue=("revenue","sum"),
            profit=("profit","sum")
        )
        .sort_index()
    )

    growth = yearly["revenue"].pct_change() * 100

    latest_growth = growth.iloc[-1]

    if latest_growth < 0:
        diagnostics.append(
            f"Revenue declined {abs(latest_growth):.1f}% last year."
        )
    else:
        diagnostics.append(
            f"Revenue grew {latest_growth:.1f}% last year."
        )

    product_rev = (
        df.groupby("product_id")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    top_share = (
        product_rev.iloc[0]
        / product_rev.sum()
    ) * 100

    diagnostics.append(
        f"Top product contributes {top_share:.1f}% of total revenue."
    )

    monthly_avg = (
        df.groupby(df["order_date"].dt.month)
        ["revenue"]
        .mean()
    )

    diagnostics.append(
        f"Strongest month is {monthly_avg.idxmax()}."
    )

    diagnostics.append(
        f"Weakest month is {monthly_avg.idxmin()}."
    )

    top3_share = (
    product_rev.head(3).sum()
    / product_rev.sum()
    ) * 100

    diagnostics.append(
    f"Top 3 products contribute {top3_share:.1f}% of total revenue."
    )
    return diagnostics

STRATEGY_KEYWORDS = [
    "plan",
    "strategy",
    "grow",
    "growth",
    "improve",
    "expand",
    "increase revenue",
    "roadmap",
    "خطة",
    "استراتيجية",
    "ازاي",
    "احسن",
    "ازود",
    "اكبر"
]
def is_strategy_request(text):

    text = text.lower()

    return any(
        keyword in text
        for keyword in STRATEGY_KEYWORDS
    )
    
def build_strategy_context(df):

    diagnostics = generate_business_diagnostics(df)

    revenue = df["revenue"].sum()

    projected_target = revenue * 1.05

    return f"""
Business Diagnostics

{chr(10).join(diagnostics)}

Target Revenue (+5%):
{projected_target:,.0f} EGP

When generating a plan:

1. Explain findings.
2. Explain risks.
3. Explain opportunities.
4. Create a 90-day plan.
5. Create a 12-month plan.
6. Estimate impact.
7. Never give generic advice.
"""
# ── 6. SYSTEM PROMPT ─────────────────────────────────────────
SYSTEM_PROMPT = f"""
You are FUSE AI — a senior business advisor embedded inside this company.
You have an MBA-level grasp of strategy, finance, pricing, operations, and growth,
AND you have complete visibility into this business's data.

You are NOT a chatbot that summarizes data. You are a thinking advisor who reads
the data, spots what it means, and tells the owner what to DO about it.

════════════════════════════════════════════════
BUSINESS INTELLIGENCE (your foundation — know this cold)
════════════════════════════════════════════════
{data_summary}
════════════════════════════════════════════════

━━━ LANGUAGE PROTOCOL ━━━
- Mirror the user's EXACT language and script. No exceptions. Ever.
- English → English only. No Arabic, no other scripts.
- Egyptian colloquial Arabic (عامية) → write like a smart Egyptian friend texting — natural, warm, zero formality. Think IN the dialect. Arabic script only.
- Modern Standard Arabic (فصحى) → formal, structured, confident. Arabic script only.
- NEVER mix languages or scripts unless the user does first.
- CRITICAL: Your response must contain ONLY characters from the user's language script + numbers + "EGP". 
  Zero tolerance for stray characters from other languages (no Latin in Arabic responses, no Arabic in English responses, absolutely no Chinese/CJK characters under any circumstances).

━━━ HOW TO ANSWER — THE CONSULTANT STANDARD ━━━

1. ANCHOR IN DATA FIRST.
   Open with the most relevant hard number from the data. No fluff opener.
   Example: "Your revenue grew 34% from 2023 to 2024 — the business has real momentum."

2. DIAGNOSE WHAT THE DATA IS TELLING YOU.
   Don't just state figures. Read them. What's the pattern? What's the signal?
   Example: P007 leads revenue but lags on margin — that tells you something about pricing power.

3. APPLY BUSINESS EXPERTISE.
   Layer in the "so what" — pricing strategy, product mix, seasonality plays, customer concentration risk, etc.
   Speak with authority. You've advised businesses before. This isn't your first rodeo.

4. FOR FUTURE QUESTIONS (next year, projections, etc.):
   You MUST extrapolate from the trend. Calculate or estimate the trajectory.
   If revenue grew 20% YoY on average → project that forward. Say so explicitly.
   Give the number. Then say what they need to do to hit it or beat it.
   NEVER say "I don't have future data." You're a forward-looking advisor.

5. CLOSE WITH ONE SHARP ACTION.
   One concrete, specific next step. Not "improve marketing."
   Something like: "Double down on P007's volume in Q3 — that's your highest-revenue window."

━━━ ANTI-PATTERNS — NEVER DO THESE ━━━
- NEVER repeat the same point twice in different words.
- NEVER open by echoing the user's question back at them (e.g. don't start with "ازاي اعلي الارباح؟").
- NEVER use filler openers like "أول حاجة" followed by an obvious statement.
- NEVER use "أنا بعتقد" / "I believe" / "I recommend" as a crutch — state things directly with confidence.
- NEVER pad responses with obvious restatements of the question.
- One idea → one sentence. Say it once. Move on.
- No walls of bullets. Flowing, punchy paragraphs.

━━━ HONESTY & PRECISION RULES ━━━
- Real numbers from the data: state them confidently.
- Projections: label them ("At current trajectory...", "If growth holds at X%...").
- Inferences from business logic: label them ("In businesses like yours, this typically means...").
- NEVER fabricate a figure. NEVER present a guess as a fact.
- If the data has a gap, say so briefly — then advise anyway.

━━━ TONE ━━━
- Direct. Sharp. Confident. Like a senior partner, not an intern summarizing a spreadsheet.
- Monetary values always in EGP.
- Conversational length: enough to actually help, not padded.

WHEN USER ASKS FOR A PLAN:

Do not give generic advice.

Always generate:

EXECUTIVE SUMMARY

KEY FINDINGS

BIGGEST RISKS

BIGGEST OPPORTUNITIES

90-DAY ACTION PLAN

MONTH 1
MONTH 2
MONTH 3

12-MONTH GROWTH PLAN

EXPECTED IMPACT

Every recommendation must be justified using the business data.
━━━ OUT OF SCOPE ━━━
Only deflect if the question has ZERO business connection (weather, recipes, personal life).
- EN:  "That's outside my scope — I'm here to help with your business."
- EGY: "ده برا نطاقي — أنا هنا عشان أساعدك في شغلك."
- MSA: "هذا خارج نطاق عملي — أنا هنا لمساعدتك في أعمالك."
Everything else — engage.
"""

# ── 7. GREETING DETECTION ────────────────────────────────────
GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hey|hello|howdy|sup|what'?s up|yo|hiya|good\s*(morning|afternoon|evening)|"
    r"مرحبا|هاي|هلو|السلام عليكم|صباح الخير|مساء الخير|هاى|اهلا|أهلا|ازيك|ازيكم|"
    r"إيه الأخبار|إيه الأخبار|ايه الاخبار)\s*[!?.]*\s*$",
    re.IGNORECASE
)

def is_greeting(text: str) -> bool:
    return bool(GREETING_PATTERNS.match(text.strip()))

from langdetect import detect

def detect_language(text):
    try:
        lang = detect(text)

        if lang == "ar":
            return "ar"

        return "en"

    except:
        return "en"
    
def greeting_response(text: str) -> str:
    lang = detect_language(text)
    if lang == "ar":
        return "هلا! أنا FUSE AI، مستشارك التجاري. إيه اللي تحب تعرفه عن شغلك؟"
    return "Hey! I'm FUSE AI, your business advisor. What would you like to know about your business?"

# ── 8. SMART RETRIEVAL ───────────────────────────────────────
def get_relevant_context(query, n_results=25):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    retrieved = "\n".join(results["documents"][0])

    yearly_anchor = "\n".join(
        f"Year {int(r['year'])}: Revenue={r['revenue']:,.0f} EGP, "
        f"Profit={r['profit']:,.0f} EGP, Margin={r['margin']:.1%}, "
        f"YoY Rev Growth={r['rev_growth']:+.1f}%"
        for _, r in yearly.iterrows()
        if not np.isnan(r.get("rev_growth", float("nan"))) or True
    )

    return f"[Yearly Anchors]\n{yearly_anchor}\n\n[Retrieved Context]\n{retrieved}"

# ── 9. GROQ CLIENT ───────────────────────────────────────────
# ⚠️  Move your API key to an environment variable:
#     set GROQ_API_KEY=your_key_here  (Windows)
#     export GROQ_API_KEY=your_key_here  (Mac/Linux)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key= os.environ.get("GROQ_API_KEY", "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")
)

# ── 10. CHAT ─────────────────────────────────────────────────
conversation_history = []

def chat_with_fuse(user_message):
    # Short-circuit greetings — no LLM call needed
    if is_greeting(user_message):
        reply = greeting_response(user_message)
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": reply})
        return reply
    
    lang = detect_language(user_message)

    if lang == "ar":
        language_system = """
    Respond ONLY in Egyptian Arabic.

    Rules:
    - Arabic script only.
    - No English words.
    - Sound like an experienced Egyptian business consultant.
    """
    else:
        language_system = """
    Respond ONLY in English.
    """

    context = get_relevant_context(user_message)
    strategy_context = ""

    if is_strategy_request(user_message):
        strategy_context = build_strategy_context(df)
    recent_history = conversation_history[-6:]

    messages = [
    {"role": "system", "content": language_system},
    {"role": "system", "content": SYSTEM_PROMPT},
    *recent_history,
    {
        "role": "user",
        "content": (
            f"Question: {user_message}\n\n"
            f"Relevant Business Data:\n{context}\n\n"
            f"{strategy_context}"
        )
    }
]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=1024
    )

    reply = response.choices[0].message.content

    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": reply})

    return reply

# ── 11. RUN ──────────────────────────────────────────────────
print("=" * 50)
print("  FUSE AI — Business Advisor")
print("=" * 50)

while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye.")
        break
    print("\nFUSE AI:", end=" ")
    print(chat_with_fuse(user_input))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    