from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter

from cleanData import mergedTable


# Constant
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "charts"
OUTPUT_DIRECTORY.mkdir(exist_ok=True)

AGE_GROUP = ["<30", "30–49", "50–69", "70+"]
ANNUAL_INCOME = ["<$30K","$30K–$60K","$60K–$100K","$100K+"]
CREDIT_SCORE = [
    "Poor: 300–579",
    "Fair: 580–669",
    "Good: 670–739",
    "Very good: 740–799",
    "Excellent: 800–850"
]
TRX_CHANNEL = [
    "Swipe Transaction",
    "Chip Transaction",
    "Online Transaction"
]

# Common functions
def saveChart(figure, filename):
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY / filename,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(figure)

def getCompleteYear(df):
    # To remove rows that don't have date 
    # Then group by year 
    # Then count month 1-12 only to check if year X has enough 12 months or not
    monthCount = df.dropna(subset=["date"]).groupby("year")["month"].nunique()
    
    # Return True / False --> Only years have enough 12 months are add to list for analysis 
    return monthCount[monthCount.eq(12)].index.tolist()

def analysisData(df):
    result = df.copy()
    
    # Create spending only amount. IF < 0 -> Not a spending so remain 0 
    result["purchase_amount"] = result["amount"].clip(lower=0)
    result["is_refund"] = result["amount"].lt(0)
    
    # Estimates how old the customer was when the transaction occurred
    result["transaction_age"] = (result["date"].dt.year - result["birth_year"])
    
    result["age_group"] = pd.cut(
        result["transaction_age"],
        bins=[0, 29, 49, 69, np.inf],
        labels=AGE_GROUP
    )
    
    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    
    result["income_bracket"] = pd.cut(
        result["yearly_income"],
        bins=[-np.inf, 30000, 60000, 100000, np.inf],
        labels=ANNUAL_INCOME
    )
    
    result["credit_score_group"] = pd.cut(
        result["credit_score"],
        bins=[299, 579, 669, 739, 799, 850],
        labels=CREDIT_SCORE
    )

    result["is_fraud"] = (
        result["label"]
        .str.title()
        .map({"Yes": True, "No": False})
        .astype("boolean")
    )
    
    return result

def categoryPreferenceTable(df, demographic_column, top_n=10):
    summary = (
        df.dropna(subset=[demographic_column, "merchant_category"])
        .groupby(
            [demographic_column, "merchant_category"],
            observed=True
        )
        .agg(spending=("purchase_amount", "sum"))
        .reset_index()
    )

    summary["group_total"] = (
        summary.groupby(demographic_column, observed=True)["spending"]
        .transform("sum")
    )

    summary["spending_share"] = (
        summary["spending"] / summary["group_total"].replace(0, np.nan)
    )

    # Select categories with the greatest spending across the complete data.
    top_categories = (
        summary.groupby("merchant_category")["spending"]
        .sum()
        .nlargest(top_n)
        .index
    )

    summary = summary[
        summary["merchant_category"].isin(top_categories)
    ]

    return summary.pivot(
        index="merchant_category",
        columns=demographic_column,
        values="spending_share"
    ).fillna(0)

# Declare variables
analysisTable = analysisData(mergedTable)
completeYears = getCompleteYear(analysisTable)

#Analyse 
def q1_spendingTrendByAgeGroup(df):
    summary =(
        df[df["year"].isin(completeYears) & df["age_group"].notna()]
            .groupby(["year", "age_group"], observed=True)
            .agg(
                total_spending=("purchase_amount", "sum"),
                customers=("user_id", "nunique"),
                transactions=("transaction_id", "size")
            )
            .reset_index()
    )
    
    # Calculate spending per customer
    summary["spending_per_customer"] = (summary["total_spending"]/summary["customers"].replace(0, np.nan))
    
    # Create chart 
    figure, axis = plt.subplots(figsize=(13, 7))
    
    sns.lineplot(data=summary,
            x="year",
            y="spending_per_customer",
            hue="age_group",
            hue_order=AGE_GROUP,
            marker="o",
            linewidth=2.2,
            ax=axis
        )
    
    axis.set(
        title="Q1: Annual spending per customer by age group",
        xlabel="Complete calendar year",
        ylabel="Spending per customer"
    )
    
    saveChart(figure, "q1_spendingTrendByAgeGroup.png")
    
def q2_spendingByIncome(df):
    summary = (
        df.dropna(subset=["income_bracket"])
        .groupby("income_bracket", observed=True)
        .agg(
            total_spending=("purchase_amount", "sum"),
            customers=("user_id", "nunique"),
            transactions=("transaction_id", "size"),
            median_transaction=("purchase_amount", "median")
        )
        .reindex(ANNUAL_INCOME).reset_index()
    )
    
    # Calculate spending per customer
    summary["spending_per_customer"] = (summary["total_spending"] / summary["customers"].replace(0, np.nan))
    
    # Calculate transaction per customer
    summary["transactions_per_customer"] = ( summary["transactions"] / summary["customers"].replace(0, np.nan))

    # Compare the highest-income group with the lowest-income group.
    low_income = summary.iloc[0]
    high_income = summary.iloc[-1]

    spending_increase_pct = (
        high_income["spending_per_customer"]
        / low_income["spending_per_customer"]
        - 1
    ) * 100

    transaction_increase_pct = (
        high_income["transactions_per_customer"]
        / low_income["transactions_per_customer"]
        - 1
    ) * 100

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Create spending per customer chart
    sns.barplot(
        data=summary,
        x="income_bracket",
        y="spending_per_customer",
        hue="income_bracket",
        palette="Blues_d",
        legend=False,
        ax=axes[0]
    )

    axes[0].set(
        title=(
            "Spending per customer\n"
            f"100K+ is {spending_increase_pct:.0f}% higher than <30K"
        ),
        xlabel="Yearly income bracket",
        ylabel="Spending per customer"
    )
    
    # Create transaction percustomer chart
    sns.barplot(
        data=summary,
        x="income_bracket",
        y="transactions_per_customer",
        hue="income_bracket",
        palette="Greens_d",
        legend=False,
        ax=axes[1]
    )

    axes[1].set(
        title=(
            "Transactions per customer\n"
            f"100K+ is {transaction_increase_pct:.0f}% higher than <30K"
        ),
        xlabel="Yearly income bracket",
        ylabel="Transactions per customer"
    )
    
    figure.suptitle(
        "Q2: Spending behavior by income bracket",
        fontsize=17,
        fontweight="bold"
    )

    saveChart(figure, "q2_spendingByIncome.png")

def q3_categoryPreferences(df):
    age_table = categoryPreferenceTable(
        df,
        demographic_column="age_group",
        top_n=10
    ).reindex(columns=AGE_GROUP)

    income_table = categoryPreferenceTable(
        df,
        demographic_column="income_bracket",
        top_n=10
    ).reindex(columns=ANNUAL_INCOME)

    gender_table = categoryPreferenceTable(
        df,
        demographic_column="gender",
        top_n=10
    )

    age_table = age_table.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    income_table = income_table.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    gender_table = gender_table.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

    figure, axes = plt.subplots(1, 3, figsize=(24, 10))

    sns.heatmap(
        age_table,
        cmap="Blues",
        annot=True,
        fmt=".1%",
        cbar=False,
        ax=axes[0]
    )
    axes[0].set(
        title="Preference by age",
        xlabel="Age group",
        ylabel="Merchant category"
    )

    sns.heatmap(
        income_table,
        cmap="Greens",
        annot=True,
        fmt=".1%",
        cbar=False,
        ax=axes[1]
    )
    axes[1].set(
        title="Preference by income",
        xlabel="Income bracket",
        ylabel="Merchant category"
    )

    sns.heatmap(
        gender_table,
        cmap="Oranges",
        annot=True,
        fmt=".1%",
        cbar=False,
        ax=axes[2]
    )
    axes[2].set(
        title="Preference by gender",
        xlabel="Gender",
        ylabel="Merchant category"
    )

    figure.suptitle(
        "Q3: Category preference by customer demographics",
        fontsize=17,
        fontweight="bold"
    )

    saveChart(figure, "q3_categoryPreferences.png")

def q4a_customersByCreditScore(df):
    summary = (
        df.dropna(subset=["credit_score_group", "user_id"])
        .groupby("credit_score_group", observed=True)
        .agg(customers=("user_id", "nunique"))
        .reindex(CREDIT_SCORE)
        .reset_index()
    )

    figure, axis = plt.subplots(figsize=(12, 7))

    sns.barplot(
        data=summary,
        x="credit_score_group",
        y="customers",
        hue="credit_score_group",
        palette="Blues_d",
        legend=False,
        ax=axis
    )

    axis.set(
        title="Total customers by credit score",
        xlabel="Credit score group",
        ylabel="Number of unique customers"
    )
    axis.tick_params(axis="x", rotation=30)
    axis.ticklabel_format(style="plain", axis="y")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.0f", padding=3)

    saveChart(figure, "q4a_customersByCreditScore.png")
    
def q4b_creditScoreRelationships(df):
    summary = (
        df.dropna(subset=["credit_score_group"])
        .groupby("credit_score_group", observed=True)
        .agg(
            total_spending=("purchase_amount", "sum"),
            customers=("user_id", "nunique")
        )
        .reindex(CREDIT_SCORE)
        .reset_index()
    )

    # Calculate spending per customer
    summary["spending_per_customer"] = (summary["total_spending"] / summary["customers"].replace(0, np.nan))

    channel = (
        df.dropna(subset=["credit_score_group", "use_chip"])
        .groupby(["credit_score_group", "use_chip"], observed=True)
        .size().rename("transactions").reset_index()
    )

    channel["channel_share"] = (channel["transactions"] / channel.groupby("credit_score_group", observed=True)["transactions"].transform("sum"))

    channel_pivot = (
        channel.pivot(
            index="credit_score_group",
            columns="use_chip",
            values="channel_share"
        )
        .reindex(CREDIT_SCORE).fillna(0)
    )

    fraud = (
        df.dropna(subset=["credit_score_group", "is_fraud"])
        .groupby("credit_score_group", observed=True)
        .agg(
            transactions=("transaction_id", "size"),
            fraud_transactions=("is_fraud", "sum")
        )
        .reindex(CREDIT_SCORE)
        .reset_index()
    )

    # Calculate fraud rate
    fraud["fraud_rate"] = (
        fraud["fraud_transactions"]
        / fraud["transactions"].replace(0, np.nan)
    )

    figure, axes = plt.subplots(1, 3, figsize=(21, 7))

    sns.barplot(
        data=summary,
        x="credit_score_group",
        y="spending_per_customer",
        color="#2A9D8F",
        ax=axes[0]
    )
    axes[0].set(
        title="Spending per customer",
        xlabel="Credit score",
        ylabel="Spending per customer"
    )
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].ticklabel_format(style="plain", axis="y")

    channel_pivot.plot(
        kind="bar",
        stacked=True,
        color=["#E76F51", "#2A9D8F", "#457B9D"],
        ax=axes[1]
    )
    axes[1].set(
        title="Channel choice",
        xlabel="Credit score",
        ylabel="Share of transactions"
    )
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(title="Channel", fontsize=8)

    sns.barplot(
        data=fraud,
        x="credit_score_group",
        y="fraud_transactions",
        color="#E76F51",
        ax=axes[2]
    )
    axes[2].set(
        title="Fraud cases",
        xlabel="Credit score",
        ylabel="Number of fraud cases"
    )
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].ticklabel_format(style="plain", axis="y")
    axes[2].bar_label(
        axes[2].containers[0],
        fmt="%.0f",
        padding=3
    )

    figure.suptitle(
        "Q4: Credit score relationship with spending, channel, and fraud",
        fontsize=17,
        fontweight="bold"
    )

    saveChart(figure, "q4b_creditScoreRelationships.png")

def q4c_fraudFor1000Customers(df):
    labeled = df.dropna(
        subset=["credit_score_group", "user_id", "is_fraud"]
    )

    # First summarize at customer level. This prevents customers with many
    # transactions from being counted as multiple customers.
    customer_fraud = (
        labeled.groupby(
            ["credit_score_group", "user_id"],
            observed=True
        )
        .agg(
            fraud_cases=("is_fraud", "sum"),
            has_fraud=("is_fraud", "max")
        )
        .reset_index()
    )

    summary = (
        customer_fraud.groupby("credit_score_group", observed=True)
        .agg(
            customers=("user_id", "nunique"),
            fraud_cases=("fraud_cases", "sum"),
            customers_with_fraud=("has_fraud", "sum")
        )
        .reindex(CREDIT_SCORE)
        .reset_index()
    )

    summary["fraud_cases_per_1000_customers"] = (
        summary["fraud_cases"]
        / summary["customers"].replace(0, np.nan)
        * 1_000
    )

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(
        data=summary,
        x="credit_score_group",
        y="customers",
        color="#457B9D",
        ax=axes[0]
    )
    axes[0].set(
        title="Customers with known fraud labels",
        xlabel="Credit score",
        ylabel="Number of customers"
    )

    sns.barplot(
        data=summary,
        x="credit_score_group",
        y="fraud_cases_per_1000_customers",
        color="#E76F51",
        ax=axes[1]
    )
    axes[1].set(
        title="Fraud cases adjusted for group size",
        xlabel="Credit score",
        ylabel="Fraud cases per 1,000 customers"
    )

    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.ticklabel_format(style="plain", axis="y")
        axis.bar_label(axis.containers[0], fmt="%.1f", padding=3)

    figure.suptitle(
        "Q4b: Fraud by credit score, adjusted for customer population",
        fontsize=17,
        fontweight="bold"
    )

    saveChart(figure, "q4b_fraudAdjustedForCustomers.png")

def q5_riskiestCategories(df):
    summary = (
        df.dropna(subset=["merchant_category", "is_fraud"])
        .groupby("merchant_category", observed=True)
        .agg(
            transactions=("transaction_id", "size"),
            fraud_transactions=("is_fraud", "sum")
        )
        .reset_index()
    )
    
    # Calculate fraud rate
    summary["fraud_rate"] = (summary["fraud_transactions"]/ summary["transactions"].replace(0, np.nan))

    # Prevent tiny categories from appearing riskiest by chance.
    summary = summary[summary["transactions"] >= 1_000]

    riskiest = (
        summary.nlargest(15, "fraud_rate")
        .sort_values("fraud_rate")
    )

    figure, axis = plt.subplots(figsize=(12, 8))

    sns.barplot(
        data=riskiest,
        x="fraud_rate",
        y="merchant_category",
        color="#DCEDC2",
        ax=axis
    )

    axis.set(
        title="Q5: Riskiest merchant categories",
        xlabel="Fraud rate",
        ylabel="Merchant category"
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1))

    saveChart(figure, "q5_riskiestCategories.png")


def main():
    q1_spendingTrendByAgeGroup(analysisTable)
    q2_spendingByIncome(analysisTable)
    q3_categoryPreferences(analysisTable)
    q4a_customersByCreditScore(analysisTable)
    q4b_creditScoreRelationships(analysisTable)
    q4c_fraudFor1000Customers(analysisTable)
    q5_riskiestCategories(analysisTable)
    
    
if __name__ == "__main__":
    main()
