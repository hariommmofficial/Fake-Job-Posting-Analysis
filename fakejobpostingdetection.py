import os
import re
import sqlite3
import textwrap
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

def safe_div(numerator, denominator, fallback=0.0):
    """Divide numerator / denominator; return fallback if denominator is 0."""
    try:
        if denominator == 0 or pd.isna(denominator):
            return fallback
        return numerator / denominator
    except Exception:
        return fallback

def safe_1_in(pct):
    """Return '1 in N' string from a percentage, handles zero gracefully."""
    pct = float(pct)
    if pct <= 0:
        return "N/A"
    return f"{round(100 / pct):.0f}"

CSV_PATH    = r"C:\Users\sande\Videos\Fake Job Analysis\fake_job_postings.csv"
EXPORT_DIR  = r"C:\Users\sande\Videos\Fake Job Analysis\exports"
os.makedirs(EXPORT_DIR, exist_ok=True)  # create exports folder if it does not exist

REAL_COLOR  = "#2196F3"   
FAKE_COLOR  = "#F44336"  
ACCENT      = "#FF9800"  
PURPLE      = "#7C4DFF"   
GREEN       = "#43A047"   
BG          = "#F8F9FA"   

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : BG,
    "axes.grid"        : True,
    "grid.color"       : "#E0E0E0",
    "grid.linewidth"   : 0.7,
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.labelsize"   : 11,
    "axes.titlesize"   : 13,
    "xtick.labelsize"  : 10,
    "ytick.labelsize"  : 10,
    "legend.fontsize"  : 10,
})

def label_bars_h(ax, fmt="{:.1f}%", offset_frac=0.01):
    """Annotate horizontal bars with their values."""
    max_val = max(bar.get_width() for bar in ax.patches) or 1
    for bar in ax.patches:
        w = bar.get_width()
        if w > 0:
            ax.text(
                w + max_val * offset_frac,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(w),
                va="center", ha="left", fontsize=9
            )

def label_bars_v(ax, fmt="{:.1f}", offset_frac=0.015):
    """Annotate vertical bars with their values."""
    max_val = max(bar.get_height() for bar in ax.patches) or 1
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max_val * offset_frac,
                fmt.format(h),
                ha="center", va="bottom", fontsize=9
            )


print("  FAKE JOB POSTING DETECTION")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"\nCSV not found at:\n     {CSV_PATH}\n\n"
        r"     C:\\Users\\sande\\Videos\\Fake Job Analysis\\" + "\\n"
        "  ➜  Download: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction"
    )

df = pd.read_csv(CSV_PATH)
print(f"  Loaded dataset  |  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

df.rename(columns={"fraudulent": "is_fake"}, inplace=True)


# FEATURE ENGINEERING
print("\nFeature Engineering")

df["country"] = df["location"].fillna("").str.extract(r"^([A-Z]{2})")
df["country"].fillna("Unknown", inplace=True)

def extract_email_domain(text):
    if pd.isna(text):
        return None
    match = re.search(r"[\w.\-]+@([\w.\-]+\.[a-z]{2,})", str(text).lower())
    return match.group(1) if match else None

df["email_domain"] = df["description"].apply(extract_email_domain)

FREE_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com",
                "outlook.com", "ymail.com", "aol.com"}
df["free_email"] = df["email_domain"].apply(
    lambda d: 1 if d in FREE_DOMAINS else 0
)

df["has_salary"] = df["salary_range"].notna().astype(int)

df["desc_len"]    = df["description"].fillna("").str.len()
df["req_len"]     = df["requirements"].fillna("").str.len()
df["company_len"] = df["company_profile"].fillna("").str.len()
df["benefits_len"]= df["benefits"].fillna("").str.len()

CORE_FIELDS = ["salary_range", "company_profile", "requirements", "benefits"]
df["missing_fields"] = df[CORE_FIELDS].isna().sum(axis=1)

df["telecommute"]    = df["telecommuting"].fillna(0).astype(int)
df["has_questions"]  = df["has_questions"].fillna(0).astype(int)
df["employment_type"]= df["employment_type"].fillna("Not Specified")
df["required_edu"]   = df["required_education"].fillna("Not Specified")
df["required_exp"]   = df["required_experience"].fillna("Not Specified")

# Human-readable label column
df["label"] = df["is_fake"].map({0: "Real", 1: "Fake"})

print(f"  Real postings : {(df.is_fake == 0).sum():,}")
print(f"  Fake postings : {(df.is_fake == 1).sum():,}")
print(f"  Overall fraud rate : {df.is_fake.mean()*100:.2f}%")

print("\nBuilding SQLite in-memory database -")
conn = sqlite3.connect(":memory:")           
df.to_sql("jobs", conn, if_exists="replace", index=False)
print("  Table [jobs] created with", len(df), "rows")

def run_sql(label, sql, conn=conn, show=True):
    """Execute a SQL query, optionally print the result, return a DataFrame."""
    res = pd.read_sql_query(sql, conn)
    if show:
        print(f"\n  SQL: {label}")
        print(textwrap.indent(res.to_string(index=False), "     "))
        print("  " + "-" * 50)
    return res


print("  SQL QUERY RESULTS - ")


# Overall statistics
q1 = run_sql("Overall Fraud Rate", """
SELECT
    COUNT(*)                                          AS total_jobs,
    SUM(is_fake)                                      AS fake_jobs,
    SUM(1 - is_fake)                                  AS real_jobs,
    ROUND(100.0 * SUM(is_fake) / COUNT(*), 3)         AS fraud_rate_pct,
    ROUND(100.0 * SUM(1-is_fake) / COUNT(*), 3)       AS real_rate_pct
FROM jobs;
""")

#Fraud rate by employment type
q2 = run_sql("Fraud Rate by Employment Type", """
SELECT
    employment_type,
    COUNT(*)                                          AS total,
    SUM(is_fake)                                      AS fake,
    SUM(1 - is_fake)                                  AS real,
    ROUND(100.0 * SUM(is_fake) / COUNT(*), 2)         AS fraud_pct
FROM jobs
GROUP BY employment_type
ORDER BY fraud_pct DESC;
""")

# Top 10 countries by fake postings
q3 = run_sql("Top 10 Countries by Fake Count", """
SELECT
    country,
    COUNT(*)                                          AS total_posts,
    SUM(is_fake)                                      AS fake_count,
    SUM(1 - is_fake)                                  AS real_count,
    ROUND(100.0 * SUM(is_fake) / COUNT(*), 2)         AS fraud_pct
FROM jobs
WHERE country != 'Unknown'
GROUP BY country
HAVING total_posts > 50
ORDER BY fake_count DESC
LIMIT 10;
""")

# Missing fields breakdown
q4 = run_sql("Missing Fields by Real vs Fake", """
SELECT
    is_fake,
    missing_fields,
    COUNT(*)                                          AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY is_fake), 2)
                                                      AS pct_within_class
FROM jobs
GROUP BY is_fake, missing_fields
ORDER BY is_fake, missing_fields;
""")

#Free email domain effect
q5 = run_sql("Free Email Domain vs Fraud Rate", """
SELECT
    CASE free_email WHEN 1 THEN 'Free Domain' ELSE 'Corporate Domain' END
                                                      AS email_type,
    COUNT(*)                                          AS total,
    SUM(is_fake)                                      AS fake,
    SUM(1 - is_fake)                                  AS real,
    ROUND(100.0 * SUM(is_fake) / COUNT(*), 2)         AS fraud_pct
FROM jobs
WHERE email_domain IS NOT NULL
GROUP BY free_email;
""")

#Telecommute vs fraud
q6 = run_sql("Telecommute Flag vs Fraud", """
SELECT
    CASE telecommute WHEN 1 THEN 'Telecommute=Yes' ELSE 'Telecommute=No' END
                                                      AS telecommute_flag,
    COUNT(*)                                          AS total,
    SUM(is_fake)                                      AS fake,
    ROUND(100.0 * SUM(is_fake) / COUNT(*), 2)         AS fraud_pct
FROM jobs
GROUP BY telecommute;
""")

#Has questions vs fraud
q7 = run_sql("Pre-Screening Questions vs Fraud", """
SELECT
    CASE has_questions WHEN 1 THEN 'Has Questions' ELSE 'No Questions' END
                                                      AS questions_flag,
    COUNT(*)                                          AS total,
    SUM(is_fake)                                      AS fake,
    ROUND(100.0 * SUM(is_fake) / COUNT(*), 2)         AS fraud_pct
FROM jobs
GROUP BY has_questions;
""")

#Average text length
q8 = run_sql("Average Text Field Lengths", """
SELECT
    CASE is_fake WHEN 1 THEN 'Fake' ELSE 'Real' END   AS posting_type,
    ROUND(AVG(desc_len), 0)                           AS avg_description_chars,
    ROUND(AVG(req_len), 0)                            AS avg_requirements_chars,
    ROUND(AVG(company_len), 0)                        AS avg_company_profile_chars,
    ROUND(AVG(benefits_len), 0)                       AS avg_benefits_chars
FROM jobs
GROUP BY is_fake;
""")

# Has salary vs fraud 
q9 = run_sql("Salary Range Presence vs Fraud", """
SELECT
    CASE has_salary WHEN 1 THEN 'Salary Given' ELSE 'No Salary Info' END
                                                      AS salary_flag,
    COUNT(*)                                          AS total,
    SUM(is_fake)                                      AS fake,
    ROUND(100.0 * SUM(is_fake) / COUNT(*), 2)         AS fraud_pct
FROM jobs
GROUP BY has_salary
ORDER BY fraud_pct DESC;
""")

#Required education vs fraud
q10 = run_sql("Required Education vs Fraud Rate", """
SELECT
    required_edu,
    COUNT(*)                                          AS total,
    SUM(is_fake)                                      AS fake,
    ROUND(100.0 * SUM(is_fake) / COUNT(*), 2)         AS fraud_pct
FROM jobs
GROUP BY required_edu
HAVING total > 100
ORDER BY fraud_pct DESC;
""")

#Required experience vs fraud
q11 = run_sql("Required Experience vs Fraud Rate", """
SELECT
    required_exp,
    COUNT(*)                                          AS total,
    SUM(is_fake)                                      AS fake,
    ROUND(100.0 * SUM(is_fake) / COUNT(*), 2)         AS fraud_pct
FROM jobs
GROUP BY required_exp
HAVING total > 100
ORDER BY fraud_pct DESC;
""")

# Combo risk — telecommute + no salary 
q12 = run_sql("High-Risk Combo: Telecommute + No Salary", """
SELECT
    CASE WHEN telecommute=1 AND has_salary=0 THEN 'Remote + No Salary'
         WHEN telecommute=1 AND has_salary=1 THEN 'Remote + Has Salary'
         WHEN telecommute=0 AND has_salary=0 THEN 'On-site + No Salary'
         ELSE                                     'On-site + Has Salary'
    END                                               AS risk_profile,
    COUNT(*)                                          AS total,
    SUM(is_fake)                                      AS fake,
    ROUND(100.0 * SUM(is_fake) / COUNT(*), 2)         AS fraud_pct
FROM jobs
GROUP BY risk_profile
ORDER BY fraud_pct DESC;
""")

conn.close()



# SCAM KEYWORD ANALYSIS  (for this Python is used here )
SCAM_KEYWORDS = [
    "work from home", "earn money", "no experience required",
    "unlimited income", "be your own boss", "guaranteed",
    "weekly pay", "bonus", "part time", "training provided",
    "immediate start", "flexible hours", "100%", "passive income",
    "data entry", "click here", "apply now", "free registration",
    "wire transfer", "credit card", "advance fee",
]

fake_text = df[df.is_fake == 1]["description"].fillna("").str.lower().str.cat(sep=" ")
real_text = df[df.is_fake == 0]["description"].fillna("").str.lower().str.cat(sep=" ")

n_fake = (df.is_fake == 1).sum()
n_real = (df.is_fake == 0).sum()

kw_df = pd.DataFrame({
    "keyword"       : SCAM_KEYWORDS,
    "fake_raw"      : [fake_text.count(k) for k in SCAM_KEYWORDS],
    "real_raw"      : [real_text.count(k) for k in SCAM_KEYWORDS],
})
kw_df["fake_per1k"] = kw_df["fake_raw"] / n_fake * 1000
kw_df["real_per1k"] = kw_df["real_raw"] / n_real * 1000
kw_df["ratio"]      = (kw_df["fake_per1k"] / kw_df["real_per1k"].replace(0, 0.01)).round(1)
kw_df.sort_values("fake_per1k", ascending=True, inplace=True)



# VISUALISATIONS
print("\n")


# Plot 1 — Dataset Overview (Pie + KPI Bar)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Plot 1 — Fake Job Postings: Dataset Overview",
             fontsize=15, fontweight="bold", y=1.02)

# Pie
sizes  = [n_real, n_fake]
clrs   = [REAL_COLOR, FAKE_COLOR]
expl   = (0, 0.07)
axes[0].pie(
    sizes,
    labels      = ["Real Jobs", "Fake Jobs"],
    autopct     = "%1.2f%%",
    colors      = clrs,
    explode     = expl,
    startangle  = 90,
    textprops   = {"fontsize": 12},
    wedgeprops  = {"linewidth": 2, "edgecolor": "white"},
    shadow      = True,
)
axes[0].set_title("Overall Posting Distribution", fontweight="bold")

# KPI bars
kpis = ["Total Postings", "Real Postings", "Fake Postings", "Fraud Rate (%)"]
vals = [len(df), n_real, n_fake, round(df.is_fake.mean() * 100, 2)]
bar_colors = ["#607D8B", REAL_COLOR, FAKE_COLOR, ACCENT]
bars = axes[1].barh(kpis, vals, color=bar_colors, height=0.45, edgecolor="white")
for bar, val in zip(bars, vals):
    axes[1].text(
        bar.get_width() + max(vals) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:,}",
        va="center", fontsize=11, fontweight="bold"
    )
axes[1].set_xlim(0, max(vals) * 1.18)
axes[1].set_title("Key Performance Indicators (KPIs)", fontweight="bold")
axes[1].set_xlabel("Value")

plt.tight_layout()
fig.savefig(os.path.join(EXPORT_DIR, "01_overview.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Plot 1 saved : 01_overview.png Dataset Overview")



# Plot 2 — Fraud Rate by Employment Type
emp = q2.sort_values("fraud_pct", ascending=True)
avg_fraud = df.is_fake.mean() * 100

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Plot 2 — Fraud Rate by Employment Type",
             fontsize=15, fontweight="bold")

bar_c = [FAKE_COLOR if p > avg_fraud else REAL_COLOR for p in emp["fraud_pct"]]
ax.barh(emp["employment_type"], emp["fraud_pct"],
        color=bar_c, edgecolor="white", height=0.55)
ax.axvline(avg_fraud, color="grey", ls="--", lw=1.5, label=f"Avg fraud rate ({avg_fraud:.1f}%)")
label_bars_h(ax)
ax.set_xlabel("Fraud Rate (%)")
ax.set_title("Red = above average fraud rate | Blue = below average", fontsize=11)

# Add small count annotations inside the bars
for i, row in enumerate(emp.itertuples()):
    ax.text(0.3, i, f"n={row.total:,}", va="center", fontsize=8, color="white", fontweight="bold")

ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(EXPORT_DIR, "02_fraud_by_employment_type.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Plot 2 saved : 02_fraud_by_employment_type.png Employment Type")


# PLOT 3 — Geographical Analysis (Fake Count + Fraud Rate)
ctry = q3.copy()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Plot 3 — Geographical Analysis of Fake Job Postings",
             fontsize=15, fontweight="bold")

# Left: absolute fake counts
axes[0].barh(ctry["country"][::-1], ctry["fake_count"][::-1],
             color=FAKE_COLOR, edgecolor="white", height=0.55)
axes[0].set_xlabel("Number of Fake Postings")
axes[0].set_title("Top 10 Countries — Absolute Fake Count")
for i, row in enumerate(ctry.iloc[::-1].itertuples()):
    axes[0].text(row.fake_count + 2, i, f"{row.fake_count:,}", va="center", fontsize=9)

# Right: fraud rate percentage
axes[1].barh(ctry["country"][::-1], ctry["fraud_pct"][::-1],
             color=ACCENT, edgecolor="white", height=0.55)
axes[1].set_xlabel("Fraud Rate (%)")
axes[1].set_title("Top 10 Countries — Fraud Rate (%)")
for i, row in enumerate(ctry.iloc[::-1].itertuples()):
    axes[1].text(row.fraud_pct + 0.1, i, f"{row.fraud_pct}%", va="center", fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(EXPORT_DIR, "03_top_countries.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Plot 3 saved : 03_top_countries.png Geographical Analysis")


# PLOT 4 — Missing Fields Pattern (Grouped Bar + Heatmap)
miss = q4.copy()
miss["class"] = miss["is_fake"].map({0: "Real", 1: "Fake"})
pivot = miss.pivot(index="missing_fields", columns="class", values="pct_within_class").fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Plot 4 — Missing Fields: Structural Red Flag in Fake Postings",
             fontsize=15, fontweight="bold")

x = np.arange(len(pivot.index))
w = 0.35
axes[0].bar(x - w/2, pivot.get("Real", 0), w, color=REAL_COLOR, label="Real", edgecolor="white")
axes[0].bar(x + w/2, pivot.get("Fake", 0), w, color=FAKE_COLOR, label="Fake", edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{i} missing" for i in pivot.index])
axes[0].set_xlabel("Number of Missing Core Fields (out of 4)")
axes[0].set_ylabel("% of Postings within Class")
axes[0].set_title("% Breakdown by Missing Field Count")
axes[0].legend()

# Heatmap
sns.heatmap(pivot.T, annot=True, fmt=".1f", cmap="RdYlGn_r",
            ax=axes[1], cbar_kws={"label": "% of class"},
            linewidths=0.5, linecolor="white")
axes[1].set_title("Heatmap — Missing Field Distribution (%)")
axes[1].set_xlabel("Missing Field Count")

plt.tight_layout()
fig.savefig(os.path.join(EXPORT_DIR, "04_missing_fields.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Plot 4 saved : 04_missing_fields.png Missing Fields Pattern")


# PLOT 5 — Email Domain Analysis
email_r = q5.copy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Plot 5 — Free Email Domain Usage in Job Postings",
             fontsize=15, fontweight="bold")

x = np.arange(len(email_r))
w = 0.35
axes[0].bar(x - w/2, email_r["real"], w, color=REAL_COLOR, label="Real", edgecolor="white")
axes[0].bar(x + w/2, email_r["fake"], w, color=FAKE_COLOR, label="Fake", edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels(email_r["email_type"])
axes[0].set_ylabel("Number of Postings")
axes[0].set_title("Posting Counts by Email Domain Type")
axes[0].legend()

axes[1].bar(email_r["email_type"], email_r["fraud_pct"],
            color=[REAL_COLOR, FAKE_COLOR][:len(email_r)], edgecolor="white", width=0.4)
for i, row in enumerate(email_r.itertuples()):
    axes[1].text(i, row.fraud_pct + 0.5, f"{row.fraud_pct}%",
                 ha="center", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Fraud Rate (%)")
axes[1].set_title("Fraud Rate: Free vs Corporate Email Domain")
max_epct = max(email_r["fraud_pct"]) if max(email_r["fraud_pct"]) > 0 else 1
axes[1].set_ylim(0, max_epct * 1.25)

plt.tight_layout()
fig.savefig(os.path.join(EXPORT_DIR, "05_email_domain.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Plot 5 saved : 05_email_domain.png Email Domain Analysis")


# PLOT 6 — Text Length Distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Plot 6 — Text Field Length Distributions: Real vs Fake",
             fontsize=15, fontweight="bold")

fields_info = [
    ("desc_len",     "Job Description Length",    axes[0, 0]),
    ("req_len",      "Requirements Length",        axes[0, 1]),
    ("company_len",  "Company Profile Length",     axes[1, 0]),
    ("benefits_len", "Benefits Section Length",    axes[1, 1]),
]

for col, title, ax in fields_info:
    clip_val = df[col].quantile(0.97)
    real_vals = df[df.is_fake == 0][col].clip(0, clip_val)
    fake_vals = df[df.is_fake == 1][col].clip(0, clip_val)
    ax.hist(real_vals, bins=50, alpha=0.65, color=REAL_COLOR, label="Real", density=True)
    ax.hist(fake_vals, bins=50, alpha=0.65, color=FAKE_COLOR, label="Fake", density=True)
    # Mark medians
    ax.axvline(real_vals.median(), color=REAL_COLOR, ls="--", lw=1.5,
               label=f"Real median: {real_vals.median():.0f}")
    ax.axvline(fake_vals.median(), color=FAKE_COLOR, ls="--", lw=1.5,
               label=f"Fake median: {fake_vals.median():.0f}")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(EXPORT_DIR, "06_text_length_distribution.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Plot 6 saved : 06_text_length_distribution.png Text Length Distributions")


# PLOT 7 — Box-Plot Feature Comparison
fig, axes = plt.subplots(1, 4, figsize=(17, 5))
fig.suptitle("Plot 7 — Key Numeric Features: Real vs Fake (Box Plots)",
             fontsize=15, fontweight="bold")

box_features = [
    ("missing_fields", "Missing Fields Count"),
    ("desc_len",        "Description Length"),
    ("req_len",         "Requirements Length"),
    ("company_len",     "Company Profile Length"),
]
palette = {"Real": REAL_COLOR, "Fake": FAKE_COLOR}
flier_p = {"marker": "o", "markersize": 2, "alpha": 0.25, "color": "grey"}

for ax, (feat, title) in zip(axes, box_features):
    plot_df = df[["label", feat]].copy()
    plot_df[feat] = plot_df[feat].clip(0, df[feat].quantile(0.97))
    sns.boxplot(data=plot_df, x="label", y=feat, palette=palette,
                order=["Real", "Fake"], ax=ax, width=0.5,
                flierprops=flier_p)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Add median annotations
    for i, lbl in enumerate(["Real", "Fake"]):
        med = plot_df[plot_df.label == lbl][feat].median()
        ax.text(i, med, f" {med:.0f}", va="bottom", ha="center",
                fontsize=9, color="black", fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(EXPORT_DIR, "07_feature_boxplots.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Plot 7 saved : 07_feature_boxplots.png Box Plots")


# PLOT 8 — Behavioural Flags (Salary, Telecommute, Questions)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Plot 8 — Behavioural Flag Analysis",
             fontsize=15, fontweight="bold")

flag_data = [
    (q9,  "salary_flag",       "Salary Range Presence"),
    (q6,  "telecommute_flag",  "Telecommute Flag"),
    (q7,  "questions_flag",    "Pre-Screening Questions"),
]

for ax, (res, col, title) in zip(axes, flag_data):
    x = np.arange(len(res))
    ax.bar(x, res["total"],  color="#B0BEC5", label="Total",     width=0.45)
    ax.bar(x, res["fake"],   color=FAKE_COLOR, label="Fake Jobs", width=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(res[col].values, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)

    # Overlay fraud rate on twin y-axis
    ax2 = ax.twinx()
    ax2.plot(x, res["fraud_pct"], "o--", color=ACCENT, lw=2.5, ms=9)
    for xi, fp in zip(x, res["fraud_pct"].values):
        ax2.text(xi, fp + 0.5, f"{fp}%", ha="center", fontsize=9,
                 fontweight="bold", color=ACCENT)
    ax2.set_ylabel("Fraud Rate (%)", color=ACCENT)
    ax2.tick_params(axis="y", colors=ACCENT)
    ax2.legend(["Fraud %"], loc="upper right", fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(EXPORT_DIR, "08_flags_analysis.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Plot 8 saved : 08_flags_analysis.png Behavioural Flags")


# PLOT 9 — Scam Keyword Frequency (Per 1 000 Postings)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Plot 9 — Scam Keyword Analysis (Normalised per 1 000 Postings)",
             fontsize=15, fontweight="bold")

y = np.arange(len(kw_df))
w = 0.38

# Left panel: absolute frequency
axes[0].barh(y - w/2, kw_df["real_per1k"], w, color=REAL_COLOR, label="Real (per 1k)", edgecolor="white")
axes[0].barh(y + w/2, kw_df["fake_per1k"], w, color=FAKE_COLOR, label="Fake (per 1k)", edgecolor="white")
axes[0].set_yticks(y)
axes[0].set_yticklabels(kw_df["keyword"], fontsize=9)
axes[0].set_xlabel("Keyword Frequency (per 1,000 postings)")
axes[0].set_title("Normalised Frequency — Real vs Fake")
axes[0].legend()

# Right panel: Fake/Real ratio — how many times more common in fakes
bar_c_ratio = [FAKE_COLOR if r >= 2 else ACCENT for r in kw_df["ratio"]]
axes[1].barh(y, kw_df["ratio"], color=bar_c_ratio, edgecolor="white")
axes[1].axvline(1, color="grey", ls="--", lw=1.2, label="Ratio = 1 (equal)")
axes[1].set_yticks(y)
axes[1].set_yticklabels(kw_df["keyword"], fontsize=9)
axes[1].set_xlabel("Fake/Real Occurrence Ratio")
axes[1].set_title("How Much More Common in Fake Posts?\n(Red = 2× or more)")
axes[1].legend()
for i, (_, row) in enumerate(kw_df.iterrows()):
    if row["ratio"] > 0:
        axes[1].text(row["ratio"] + 0.05, i, f"{row['ratio']:.1f}×", va="center", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(EXPORT_DIR, "09_scam_keywords.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Plot 9 saved : 09_scam_keywords.png Scam Keywords")


# PLOT 10 — Comprehensive 5-Panel Dashboard
fig = plt.figure(figsize=(18, 11))
fig.suptitle(
    "Plot 10 — Fake Job Detection: Full Analysis Dashboard",
    fontsize=17, fontweight="bold", y=1.01
)
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

# Panel A — Fraud overview pie
ax_a = fig.add_subplot(gs[0, 0])
ax_a.pie(
    [n_real, n_fake], labels=["Real", "Fake"],
    autopct="%1.1f%%", colors=[REAL_COLOR, FAKE_COLOR],
    explode=(0, 0.07), startangle=90,
    textprops={"fontsize": 11},
    wedgeprops={"linewidth": 2, "edgecolor": "white"}
)
ax_a.set_title("A. Fraud Overview", fontweight="bold", fontsize=12)

# Panel B — Top 5 countries
ax_b = fig.add_subplot(gs[0, 1])
top5 = q3.head(5)
ax_b.barh(top5["country"][::-1], top5["fake_count"][::-1],
          color=FAKE_COLOR, edgecolor="white")
ax_b.set_xlabel("Fake Postings")
ax_b.set_title("B. Top 5 Countries", fontweight="bold", fontsize=12)
for i, row in enumerate(top5.iloc[::-1].itertuples()):
    ax_b.text(row.fake_count + 1, i, str(row.fake_count), va="center", fontsize=9)

# Panel C — Missing fields grouped bars
ax_c = fig.add_subplot(gs[0, 2])
x_m = np.arange(5)
ax_c.bar(x_m - 0.2, [df[(df.is_fake==0)]["missing_fields"].value_counts().get(i, 0) for i in range(5)],
         0.38, color=REAL_COLOR, label="Real", edgecolor="white")
ax_c.bar(x_m + 0.2, [df[(df.is_fake==1)]["missing_fields"].value_counts().get(i, 0) for i in range(5)],
         0.38, color=FAKE_COLOR, label="Fake", edgecolor="white")
ax_c.set_xticks(x_m)
ax_c.set_xticklabels([f"{i}" for i in range(5)])
ax_c.set_xlabel("Missing Fields Count")
ax_c.set_title("C. Missing Fields Pattern", fontweight="bold", fontsize=12)
ax_c.legend(fontsize=9)

# Panel D — Top 8 scam keywords
ax_d = fig.add_subplot(gs[1, 0:2])
top_kw = kw_df.tail(8)
y_k    = np.arange(len(top_kw))
ax_d.barh(y_k - 0.2, top_kw["real_per1k"], 0.38, color=REAL_COLOR,
          label="Real (per 1k)", edgecolor="white")
ax_d.barh(y_k + 0.2, top_kw["fake_per1k"], 0.38, color=FAKE_COLOR,
          label="Fake (per 1k)", edgecolor="white")
ax_d.set_yticks(y_k)
ax_d.set_yticklabels(top_kw["keyword"], fontsize=9)
ax_d.set_xlabel("Frequency per 1,000 postings")
ax_d.set_title("D. Top Scam Keywords", fontweight="bold", fontsize=12)
ax_d.legend(fontsize=9)

# Panel E — Risk combo fraud rates
ax_e = fig.add_subplot(gs[1, 2])
colors_e = [FAKE_COLOR if v > 15 else ACCENT for v in q12["fraud_pct"]]
ax_e.barh(q12["risk_profile"][::-1], q12["fraud_pct"][::-1],
          color=colors_e[::-1], edgecolor="white")
ax_e.set_xlabel("Fraud Rate (%)")
ax_e.set_title("E. Risk Profile Combos", fontweight="bold", fontsize=12)
for i, row in enumerate(q12.iloc[::-1].itertuples()):
    ax_e.text(row.fraud_pct + 0.3, i, f"{row.fraud_pct}%", va="center", fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(EXPORT_DIR, "10_dashboard.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Plot 10 saved : 10_dashboard.png Dashboard")

overall  = q1.iloc[0]
top_ctry = q3.iloc[0]


print()
print("  FAKE JOB POSTING DETECTION REPORT BY HARIOM")
print()
print("  Dataset     : Kaggle --- Real or Fake? Fake Job Posting Prediction")
print(f"  Total Jobs  : {int(overall['total_jobs']):,}")
print(f"  Real Jobs   : {int(overall['real_jobs']):,}  ({overall['real_rate_pct']}%)")
print(f"  Fake Jobs   : {int(overall['fake_jobs']):,}   ({overall['fraud_rate_pct']}%)")
print(f"  Fraud Rate  : {overall['fraud_rate_pct']}%  ---  1 in every {safe_1_in(overall['fraud_rate_pct'])} listings is fraudulent")
print()
print("  KEY FINDINGS")
print()
print("  1. FRAUD RATE")
print(f"     - {overall['fraud_rate_pct']}% of all postings are fraudulent.")
print("     - This is a statistically significant share --- nearly 1 in 20 jobs.")
print()
print("  2. TOP FRAUDULENT COUNTRY")
print(f"     - {top_ctry['country']} leads with {int(top_ctry['fake_count']):,} fake postings ({top_ctry['fraud_pct']}% fraud rate).")
print()
print("  3. MISSING FIELDS  (Structural Red Flag #1)")
print("     - Fake postings have far more missing core fields:")
print("       salary range, company profile, requirements, and benefits.")
print("     - Postings with 3 or 4 missing fields are overwhelmingly fraudulent.")
print()
print("  4. FREE EMAIL DOMAINS  (Red Flag #2)")
print("     - Postings with Gmail / Yahoo / Hotmail addresses show a")
print("       significantly higher fraud rate than corporate domains.")
print()
print("  5. SCAM KEYWORDS  (Red Flag #3)")
print("     - Terms such as work from home, no experience required,")
print("       earn money, and weekly pay appear far more in fake listings.")
print()
print("  6. EMPLOYMENT TYPE")
print("     - Part-time and Contract listings have above-average fraud rates.")
print()
print("  7. TELECOMMUTE POSTINGS")
print("     - Remote-enabled postings are more frequently fraudulent.")
print("       Scammers exploit high demand for remote work opportunities.")
print()
print("  8. SALARY INFORMATION")
print("     - Listings WITHOUT salary information have a higher fraud rate.")
print("       Legitimate employers are more transparent about compensation.")
print()
print("  9. HIGH-RISK COMBINATION")
print("     - Remote + No Salary is the riskiest profile combination.")
print("       Job seekers should be especially cautious of such postings.")
print()
print("  RECOMMENDATIONS")
print()
print("  1. Flag postings with 2 or more missing core fields for manual review.")
print("  2. Block or quarantine listings using free personal email domains.")
print("  3. Run NLP keyword scanning on descriptions before publishing.")
print("  4. Cross-check company profiles against official business registries.")
print("  5. Apply stricter verification for remote and no-experience postings.")
print("  6. Require salary transparency as a mandatory field on job boards.")
print()
print()
print("  Analysis complete. All 10 plots saved to the exports folder.")
print(f"  Exports folder : {EXPORT_DIR}")
print()
