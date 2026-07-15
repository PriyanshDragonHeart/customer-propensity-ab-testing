import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================
# CONFIGURATION
# ============================================

N_CLIENTS = 500000   # Change to any number

# ============================================
# CLIENT ID
# ============================================

client_code = [
    f"C{10000000+i}"
    for i in range(N_CLIENTS)
]

# ============================================
# CURRENT AGE
# ============================================

current_age = np.clip(
    np.random.normal(
        loc=39.5,
        scale=14.5,
        size=N_CLIENTS
    ),
    18,
    85
).astype(int)

# ============================================
# ENTRY AGE
# ============================================

entry_age = (
    current_age -
    np.random.randint(
        0,
        25,
        N_CLIENTS
    )
)

entry_age = np.clip(
    entry_age,
    18,
    current_age
)

# ============================================
# VINTAGE
# ============================================

vintage_months = np.random.gamma(
    shape=2.4,
    scale=32,
    size=N_CLIENTS
)

vintage_months = np.clip(
    vintage_months,
    11,
    359
).astype(int)

# ============================================
# RECENCY
# ============================================

recency = np.random.randint(
    56,
    4384,
    N_CLIENTS
).astype(float)

# About 44% clients never traded
mask = np.random.rand(N_CLIENTS) < 0.44

recency[mask] = np.nan

# ============================================
# DP VALUE
# ============================================

dp = np.random.lognormal(
    mean=7,
    sigma=2,
    size=N_CLIENTS
)

zero_mask = np.random.rand(N_CLIENTS) < 0.70

dp[zero_mask] = 0

DP_Value = np.round(dp,2)

# ============================================
# TOTAL MARGIN
# ============================================

margin = np.random.lognormal(
    mean=6,
    sigma=2,
    size=N_CLIENTS
)

margin[np.random.rand(N_CLIENTS)<0.55]=0

Total_Margin = np.round(margin,2)

# ============================================
# CASH ORDERS
# ============================================

ord_cash = np.random.poisson(
    3,
    N_CLIENTS
)

heavy = np.random.rand(N_CLIENTS)<0.05

ord_cash[heavy]+=np.random.randint(
    30,
    250,
    heavy.sum()
)

# ============================================
# DERIVATIVE ORDERS
# ============================================

ord_derv=np.random.poisson(
    1,
    N_CLIENTS
)

heavy=np.random.rand(N_CLIENTS)<0.02

ord_derv[heavy]+=np.random.randint(
    50,
    300,
    heavy.sum()
)

# ============================================
# COMMODITY
# ============================================

ord_comm=np.random.binomial(
    3,
    0.02,
    N_CLIENTS
)

# ============================================
# CURRENCY
# ============================================

ord_curr=np.random.binomial(
    2,
    0.01,
    N_CLIENTS
)

# ============================================
# CATEGORY
# ============================================

Category_Bucket_final=np.random.choice(

[
"1. Platinum",
"2. Gold",
"3. Silver",
"4. Bronze",
"5. Standard",
"6. Base"
],

size=N_CLIENTS,

p=[
0.001,
0.002,
0.002,
0.003,
0.007,
0.985
]
)

# ============================================
# VERTICAL
# ============================================

Vertical=np.random.choice(

[
"Online",
"Bank",
"Franchisee",
"Branch",
"HVC"
],

size=N_CLIENTS,

p=[
0.80,
0.06,
0.06,
0.06,
0.02
]
)

# ============================================
# SELF DEALER
# ============================================

self_dealer_status=np.random.choice(

[
"Missing",
"100% Self",
"100% Dealer",
">75% Self",
"51-75% Self",
"26-50% Self",
"<25% Self"
],

size=N_CLIENTS,

p=[
0.52,
0.28,
0.10,
0.04,
0.02,
0.025,
0.015
]
)

# ============================================
# PLAN
# ============================================

plan=np.random.choice(

[
"Free Plan",
"Others",
"Youth",
"Pro",
"Max"
],

size=N_CLIENTS,

p=[
0.48,
0.37,
0.14,
0.009,
0.001
]
)

# ============================================
# DEALING ZONE
# ============================================

Dealing_Zone=np.random.choice(

[
"West",
"North",
"South",
"East"
],

size=N_CLIENTS,

p=[
0.39,
0.29,
0.22,
0.10
]
)

# ============================================
# TRADING MEMBER
# ============================================

weights=np.ones(48)

weights[0]=45
weights[1]=15
weights[2]=10
weights[3]=8
weights[4]=5

weights=weights/weights.sum()

Trading_member_code=np.random.choice(

np.arange(48),

size=N_CLIENTS,

p=weights

)

# ============================================
# MTF
# ============================================

MTF_taken=np.random.choice(

[0,1],

size=N_CLIENTS,

p=[
0.984,
0.016
]

)

# ============================================
# TARGET
# ============================================

recency_for_score = np.where(
    np.isnan(recency),
    9999,
    recency
)

score = (
    0.00008 * Total_Margin +
    0.00002 * DP_Value +
    0.18 * ord_cash +
    0.06 * ord_derv +
    0.012 * vintage_months -
    0.00025 * recency_for_score +
    0.02 * current_age
)

# Convert score into probability
prob = 1 / (1 + np.exp(-score))

# Scale probabilities so overall conversion is around 0.5%–1%
prob = (prob - prob.min()) / (prob.max() - prob.min())
prob = prob * 0.012

target = (
    np.random.rand(N_CLIENTS) < prob
).astype(int)

df = pd.DataFrame({

    "Client_Code": client_code,

    "current_age": current_age,
    "entry_age": entry_age,
    "vintage_months": vintage_months,

    "DP_Value": DP_Value,
    "Total_Margin": Total_Margin,

    "ord_cash": ord_cash,
    "ord_derv": ord_derv,
    "ord_comm": ord_comm,
    "ord_curr": ord_curr,

    "Category_Bucket_final": Category_Bucket_final,
    "Vertical": Vertical,
    "self_dealer_status": self_dealer_status,
    "Number_of_competition_account": Trading_member_code,
    "plan": plan,
    "Dealing_Zone": Dealing_Zone,
    "MTF_taken": MTF_taken,

    "recency": recency,

    "target": target
})

# ============================================
# SAVE
# ============================================

df.to_csv(
    "data/customer_propensity_dummy_data.csv",
    index=False
)

print("="*60)
print("Dummy dataset generated successfully!")
print("="*60)
print(df.head())
print()
print(df.describe(include="all"))
print()
print("Shape :",df.shape)
print("Positive Rate :",round(df.target.mean()*100,2),"%")