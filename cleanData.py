import pandas as pd

# Load tables
dataframes = {
    "users": pd.read_csv("./data/users_data.csv"),
    "cards": pd.read_csv("./data/cards_data.csv"),
    "transactions": pd.read_csv(
        "./data/transactions_data_25pc.csv",
        parse_dates=["date"]
    ),
    "fraud": pd.read_csv("./data/train_fraud_labels.csv"),
    "mcc": pd.read_csv("./data/mcc_codes.csv")
}

# Constant
MISSING_VALUE = {"","NA", "N/A", "null", "None", "Unknown"}
GENDER = {"Male", "Female"}
CARD_BRANDS = {"Visa", "Mastercard", "Amex", "Discover" }
CARD_TYPE = {"Debit", "Debit (Prepaid)", "Credit"}
TRANSACTION_CHANNEL = {"Swipe Transaction", "Chip Transaction", "Online Transaction"}

# Columns' data type
NUMBER_COLUMNS = {
    "users": [
        "id",
        "current_age",
        "retirement_age",
        "birth_year",
        "birth_month",
        "credit_score",
        "num_credit_cards"
    ],
    "cards": [
        "id",
        "client_id",
        "num_cards_issued",
        "year_pin_last_changed"
    ],
    "transactions": [
        "id",
        "client_id",
        "card_id",
        "merchant_id",
        "zip",
        "mcc"
    ],
    "fraud": [
        "id"
    ],
    "mcc": [
        "mcc_code"
    ]
}

MONEY_COLUMNS = {
    "users": [
        "per_capita_income",
        "yearly_income",
        "total_debt"
    ],
    "cards": [
        "credit_limit"
    ],
    "transactions": [
        "amount"
    ]
}

TEXT_COLUMNS = {
    "users": [
        "gender",
        "address"
    ],
    "cards": [
        "card_brand",
        "card_type",
        "card_number",
        "expires",
        "cvv",
        "has_chip",
        "acct_open_date",
        "card_on_dark_web"
    ],
    "transactions": [
        "use_chip",
        "merchant_city",
        "merchant_state",
        "errors"
    ],
    "fraud": [
        "label"
    ],
    "mcc": [
        "merchant_category"
    ]
}

# Common functions
def fixHeaders(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\\", "", regex=False)
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df

def convertCurrency(series):
    return pd.to_numeric (
        series.astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )
    
def cleanText(series):
    return series.astype("string").str.strip()

def convertInt(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    
    return df
        
def convertCategory(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype("category")
    return df

def mergeTables():
    # Rename primary key to prevent ambiguous id columns
    usersForJoin = dataframes["users"].rename(columns={"id": "user_id"})

    cardsForJoin = (dataframes["cards"]
        .rename(columns={
            "id": "card_id",
            "client_id": "user_id"
        })
        .drop(
            columns=["card_number", "cvv"],
            errors="ignore"
        )
    )

    transactionsForJoin = (dataframes["transactions"].rename(columns={"id": "transaction_id", "client_id": "user_id"}))

    fraudForJoin = (dataframes["fraud"].rename(columns={"id": "transaction_id"}))

    mccForJoin = dataframes["mcc"] 

    mergedTable = (transactionsForJoin
                    .merge(fraudForJoin, on="transaction_id", how="left", validate="one_to_one", sort=False)
                    .merge(mccForJoin, left_on="mcc", right_on="mcc_code", how="left", validate="many_to_one", sort=False)
                    .merge(usersForJoin, on="user_id", how="left", validate="many_to_one", sort=False)
                    .merge(cardsForJoin, on=["card_id", "user_id"], how="left", validate="many_to_one", sort=False)
                    )
    return mergedTable


# Standardize data
for tableName, df in dataframes.items():
    df = fixHeaders(df)
    df = df.replace(list(MISSING_VALUE), pd.NA)
    dataframes[tableName] = df
    
# Convert all number columns into int type
for tableName, columns in NUMBER_COLUMNS.items():
    dataframes[tableName] = convertInt(dataframes[tableName], columns)
    
# Convert currency (remove $ and ,)
for tableName, columns in MONEY_COLUMNS.items():
    for column in columns:
        dataframes[tableName][column] = convertCurrency(dataframes[tableName][column])
        
    
# Clean texts
for tableName, columns in TEXT_COLUMNS.items():
    for column in columns:
        dataframes[tableName][column] = cleanText(dataframes[tableName][column])

mergedTable = mergeTables()


