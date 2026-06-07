import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv("test_predictions.csv")

print("\n=== Basic Info ===")
print(f"Total fights: {len(df)}")
print(f"Overall win rate: {df['actual'].mean():.3f}")

# ------------------------
# 1. Confidence buckets
# ------------------------
print("\n=== Confidence Buckets ===")

bins = np.arange(0.5, 1.01, 0.05)
df["bucket"] = pd.cut(df["p_model"], bins=bins)

summary = df.groupby("bucket").apply(
    lambda x: pd.Series({
        "count": len(x),
        "avg_prob": x["p_model"].mean(),
        "win_rate": x["actual"].mean()
    })
).reset_index()

print(summary)

# ------------------------
# 2. Threshold performance
# ------------------------
print("\n=== Threshold Performance ===")

thresholds = [0.55, 0.6, 0.65, 0.7, 0.75]

for t in thresholds:
    subset = df[df["p_model"] >= t]
    if len(subset) == 0:
        continue

    win_rate = subset["actual"].mean()

    print(f"Threshold {t}:")
    print(f"  Bets: {len(subset)}")
    print(f"  Win rate: {win_rate:.3f}")
    print("")

# ------------------------
# 3. Underdog side (optional)
# ------------------------
print("\n=== Underdog Predictions ===")

underdogs = df[df["p_model"] < 0.5]

if len(underdogs) > 0:
    print(f"Total underdog picks: {len(underdogs)}")
    print(f"Win rate: {underdogs['actual'].mean():.3f}")
else:
    print("No underdog predictions found.")