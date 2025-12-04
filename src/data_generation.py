# src/data_generation.py

import numpy as np
import pandas as pd
from typing import Tuple, List

# -----------------------------
# Global config (can move to a config file later)
# -----------------------------

NUM_ACCOUNTS = 5000
MULE_FRACTION = 0.03        # 3% mule accounts
SIM_DAYS = 90               # simulate 90 days
THRESHOLD = 10_000          # USD – structuring threshold

RNG = np.random.default_rng(seed=42)


# -----------------------------
# 1. Generate accounts table
# -----------------------------

def generate_accounts(
    num_accounts: int = NUM_ACCOUNTS,
    mule_fraction: float = MULE_FRACTION,
) -> pd.DataFrame:
    """
    Create the accounts table with:
      - account_id
      - is_mule
      - typology_type
      - account_open_day
    """
    account_ids = np.arange(num_accounts)

    num_mules = int(num_accounts * mule_fraction)
    mule_indices = RNG.choice(account_ids, size=num_mules, replace=False)

    is_mule = np.isin(account_ids, mule_indices).astype(int)

    # For now, assign typology_type = "normal" for all;
    # we will overwrite for mule accounts in a separate step.
    typology_type = np.array(["normal"] * num_accounts, dtype=object)

    account_open_day = RNG.integers(low=0, high=SIM_DAYS // 3, size=num_accounts)

    accounts = pd.DataFrame({
        "account_id": account_ids,
        "is_mule": is_mule,
        "typology_type": typology_type,
        "account_open_day": account_open_day,
    })

    return accounts


# -----------------------------
# 2. Transaction generators per typology
# -----------------------------

def generate_transactions_fan_in(
    accounts: pd.DataFrame,
    n_patterns: int = 30,
    min_senders: int = 10,
    max_senders: int = 50,
    min_amount: float = 1000,
    max_amount: float = 5000,
) -> pd.DataFrame:
    """
    Fan-in typology:
    Many normal accounts send money to one mule account.
    """

    print("=== FAN-IN DEBUG START ===")
    print("Total mule accounts:", accounts[accounts["is_mule"] == 1].shape[0])
    print("Total normal accounts:", accounts[accounts["is_mule"] == 0].shape[0])

    rows = []

    # 1. Select mule collector accounts
    mule_accounts = accounts[accounts["is_mule"] == 1]["account_id"].sample(
        n=n_patterns, replace=True
    ).tolist()

    print("Mule collector sample:", mule_accounts[:5])

    for i, dst in enumerate(mule_accounts):

        senders = accounts[accounts["is_mule"] == 0]["account_id"].sample(
            RNG.integers(min_senders, max_senders)
        ).tolist()

        print(f"\nPattern {i+1}: dst={dst}, senders_count={len(senders)}")

        for src in senders:
            timestamp_day = RNG.integers(0, SIM_DAYS)
            amount = RNG.uniform(min_amount, max_amount)

            rows.append([
                None,
                timestamp_day,
                src,
                dst,
                round(amount, 2)
            ])

            # Print first few for visibility
            if len(rows) < 5:
                print("  TX:", rows[-1])

    print("\nDEBUG: Total rows generated:", len(rows))
    print("=== FAN-IN DEBUG END ===\n")

    return pd.DataFrame(
        rows,
        columns=["txn_id", "timestamp_day", "src_account_id", "dst_account_id", "amount"]
    )


import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_transactions_fan_out(
    accounts: pd.DataFrame,
    n_patterns: int = 30,
    min_receivers: int = 10,
    max_receivers: int = 50,
    min_amount: float = 1000,
    max_amount: float = 5000,
) -> pd.DataFrame:
    """
    Fan-out typology:
    One mule account sends money to many normal accounts.
    """

    print("=== FAN-OUT DEBUG START ===")
    print("Total mule accounts:", accounts[accounts["is_mule"] == 1].shape[0])
    print("Total normal accounts:", accounts[accounts["is_mule"] == 0].shape[0])

    rows = []

    # 1. Select mule sender accounts (opposite of fan-in)
    mule_senders = accounts[accounts["is_mule"] == 1]["account_id"].sample(
        n=n_patterns, replace=True
    ).tolist()

    print("Mule sender sample:", mule_senders[:5])

    for i, src in enumerate(mule_senders):

        # receivers must be NORMAL accounts (not mules)
        receivers = accounts[accounts["is_mule"] == 0]["account_id"].sample(
            RNG.integers(min_receivers, max_receivers)
        ).tolist()

        print(f"\nPattern {i+1}: src={src}, receivers_count={len(receivers)}")

        for dst in receivers:
            timestamp_day = RNG.integers(0, SIM_DAYS)
            amount = RNG.uniform(min_amount, max_amount)

            rows.append([
                None,           # txn_id placeholder
                timestamp_day,
                src,            # mule as sender
                dst,            # normal as receiver
                round(amount, 2)
            ])

            # Print first few for visibility
            if len(rows) < 5:
                print("  TX:", rows[-1])

    print("\nDEBUG (fan-out): Total rows generated:", len(rows))
    print("=== FAN-OUT DEBUG END ===\n")

    return pd.DataFrame(
        rows,
        columns=["txn_id", "timestamp_day", "src_account_id", "dst_account_id", "amount"]
    )


def generate_transactions_velocity(
    accounts: pd.DataFrame,
    n_patterns: int = 20,
) -> pd.DataFrame:
    """
    Generate rapid in–out typology:
    money comes in and leaves quickly within short windows (high velocity).
    """
    rows = []
    # TODO: implement bursty in-out flows for selected mule accounts
    return pd.DataFrame(rows, columns=["txn_id", "timestamp_day",
                                       "src_account_id", "dst_account_id", "amount"])


def generate_transactions_structuring(
    accounts: pd.DataFrame,
    n_patterns: int = 20,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """
    Generate structuring / smurfing:
    many incoming deposits just below the reporting threshold.
    """
    rows = []
    # TODO: generate deposits with amounts in [0.8*threshold, threshold - small_margin]
    return pd.DataFrame(rows, columns=["txn_id", "timestamp_day",
                                       "src_account_id", "dst_account_id", "amount"])


def generate_transactions_layering(
    accounts: pd.DataFrame,
    n_patterns: int = 10,
) -> pd.DataFrame:
    """
    Generate circular layering typology:
    chains A -> B -> C -> ... -> A.
    """
    rows = []
    # TODO: pick small rings of accounts and move funds around them in loops
    return pd.DataFrame(rows, columns=["txn_id", "timestamp_day",
                                       "src_account_id", "dst_account_id", "amount"])


def generate_normal_transactions(
    accounts: pd.DataFrame,
    avg_txn_per_account: int = 50,
) -> pd.DataFrame:
    """
    Generate background 'normal' activity for all accounts,
    with mild randomness and fewer counterparties.
    """
    rows = []
    # TODO: for each account, sample N transactions with random day, amount, counterparty
    return pd.DataFrame(rows, columns=["txn_id", "timestamp_day",
                                       "src_account_id", "dst_account_id", "amount"])


# -----------------------------
# 3. Orchestrator
# -----------------------------

def generate_full_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    High-level function:
      1. generate accounts
      2. generate normal + typology-specific transactions
      3. concatenate all transactions
      4. ensure txn_id is unique
    Returns:
      accounts_df, transactions_df
    """
    accounts = generate_accounts()

    # For now, placeholder empty frames – we will fill logic step by step
    fan_in_txn = generate_transactions_fan_in(accounts)
    fan_out_txn = generate_transactions_fan_out(accounts)
    velocity_txn = generate_transactions_velocity(accounts)
    structuring_txn = generate_transactions_structuring(accounts)
    layering_txn = generate_transactions_layering(accounts)
    normal_txn = generate_normal_transactions(accounts)

    transactions = pd.concat(
        [fan_in_txn, fan_out_txn, velocity_txn, structuring_txn, layering_txn, normal_txn],
        ignore_index=True
    )

    # Assign unique txn_id if not done already
    transactions["txn_id"] = np.arange(len(transactions))

    return accounts, transactions


if __name__ == "__main__":
    accounts_df, transactions_df = generate_full_dataset()
    print(accounts_df.head())
    print(transactions_df.head())