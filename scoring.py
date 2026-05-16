from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

def add_factor_score(df, factor_weights=None):

    positive_factors = [
        "F_REV_GROWTH_YOY", "F_EPS_GROWTH_YOY", "F_EBITDA_GROWTH_YOY",
        "F_EPS_QOQ_TREND", "F_ROE", "F_ROIC", "F_OP_MARGIN", "F_NET_MARGIN",
        "F_ASSET_TURNOVER", "F_FCF_PER_SHARE", "F_EARNINGS_YIELD",
        "MOMENTUM_240", "MOMENTUM_80"
    ]

    negative_factors = [
        "F_NET_DEBT_EBITDA", "F_CASH_CONV_CYCLE",
        "F_P_BV", "F_P_S", "STD_56D"
    ]

    all_factors = positive_factors + negative_factors

    # Default weight = 1.0
    if factor_weights is None:
        factor_weights = {}

    w = Window.partitionBy("DATE")

    # Normalize each factor with NULL → 0
    for col in all_factors:

        base_col = F.coalesce(F.col(col), F.lit(0))
        min_col = F.min(base_col).over(w)
        max_col = F.max(base_col).over(w)

        scaled = (base_col - min_col) / (max_col - min_col)

        if col in positive_factors:
            df = df.withColumn(f"N_{col}", F.coalesce(scaled, F.lit(0)))
        else:
            df = df.withColumn(f"N_{col}", F.coalesce(1 - scaled, F.lit(0)))

    # ---- Weighted score ----
    norm_cols = [f"N_{c}" for c in all_factors]

    # Build weighted sum
    weighted_sum = None
    weight_total = 0.0

    for col in all_factors:
        w_i = factor_weights.get(col, 1.0)  # default weight = 1
        weight_total += w_i

        term = F.col(f"N_{col}") * F.lit(w_i)
        weighted_sum = term if weighted_sum is None else (weighted_sum + term)

    df = df.withColumn("SCORE", weighted_sum / F.lit(weight_total))

    return df.select("TICKER", "DATE", "CLOSE", *norm_cols, "SCORE")


def assign_score_ranking(df_score):

    # Window: rank within each DATE, highest SCORE gets rank 1
    w = Window.partitionBy("DATE").orderBy(F.col("SCORE").desc())

    df_ranked = df_score.withColumn(
        "SCORE_RANK",
        F.row_number().over(w)
    )

    return df_ranked


def backtest_scoring(df_rank, strat_start_date, rank_cutoff=10, return_ahead=90):

    df_top = df_rank \
        .filter(F.col("SCORE_RANK") <= rank_cutoff) \
        .filter(F.col("DATE") >= strat_start_date)

    w_forward = (
        Window.partitionBy("TICKER")
        .orderBy("DATE")
        .rowsBetween(0, return_ahead)
    )

    df_top = df_top.withColumn(
        "FWD_CLOSE",
        F.last("CLOSE", ignorenulls=True).over(w_forward)
    )

    df_top = df_top.withColumn(
        "FWD_RETURN",
        (F.col("FWD_CLOSE") / F.col("CLOSE")) - 1
    )

    # Portfolio return per rebalance date
    df_portfolio = (
        df_top.groupBy("DATE")
        .agg(F.mean("FWD_RETURN").alias("PORTFOLIO_RETURN"))
        .orderBy("DATE")
    )

    # Keep only rebalance dates (every 90 days)
    df_portfolio = df_portfolio.withColumn(
        "ROW", F.row_number().over(Window.orderBy("DATE"))
    ).filter((F.col("ROW") - 1) % return_ahead == 0).drop("ROW")

    return df_top, df_portfolio


def save_portfolio_plot(df_portfolio, filename="portfolio_performance.png"):
    """
    Creates and saves a PNG plot of cumulative portfolio returns over time.
    df_portfolio must contain columns: DATE, PORTFOLIO_RETURN
    """

    import matplotlib.pyplot as plt
    import pandas as pd

    # Convert Spark → Pandas
    pdf = df_portfolio.orderBy("DATE").toPandas()

    # Ensure DATE is datetime
    pdf["DATE"] = pd.to_datetime(pdf["DATE"])

    # Compute cumulative return
    pdf["CUM_RETURN"] = (1 + pdf["PORTFOLIO_RETURN"]).cumprod() - 1

    # Plot
    plt.figure(figsize=(14, 6))
    # plt.plot(pdf["DATE"], pdf["PORTFOLIO_RETURN"], label="Cumulative Return", color="blue")
    plt.plot(pdf["DATE"], pdf["CUM_RETURN"], label="Cumulative Return", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Portfolio Performance Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save to PNG
    plt.savefig(filename, dpi=300)
    plt.close()

    return filename


def performance_if_started_on_each_date(df_rank, date_cutoff, rank_cutoff=10, return_ahead=90,
                                        filename="start_date_performance.png"):
    """
    For each possible start date, compute the final cumulative return of the strategy.
    Produces a PNG plot and returns a Pandas DataFrame with results.
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    df_rank = df_rank.filter(F.col("DATE") >= date_cutoff)

    # Get all unique dates sorted
    start_dates = (
        df_rank.select("DATE")
        .distinct()
        .orderBy("DATE")
        .toPandas()["DATE"]
    )

    results = []

    for start_date in tqdm(start_dates):
        # Run backtest from this start date
        _, df_portfolio = backtest_scoring(
            df_rank,
            strat_start_date=start_date,
            rank_cutoff=rank_cutoff,
            return_ahead=return_ahead
        )

        pdf = df_portfolio.orderBy("DATE").toPandas()

        if len(pdf) == 0:
            continue

        # Compute cumulative return
        pdf["CUM_RETURN"] = (1 + pdf["PORTFOLIO_RETURN"]).cumprod() - 1

        final_return = pdf["CUM_RETURN"].iloc[-1]

        results.append((start_date, final_return))

    # Convert to Pandas DataFrame
    perf_df = pd.DataFrame(results, columns=["START_DATE", "FINAL_RETURN"])
    perf_df["START_DATE"] = pd.to_datetime(perf_df["START_DATE"])

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(perf_df["START_DATE"], perf_df["FINAL_RETURN"], color="purple")
    plt.xlabel("Start Date")
    plt.ylabel("Final Return")
    plt.title("Final Strategy Performance Depending on Start Date")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return perf_df


def read_scoring_data(start_date=None, end_date=None):

    positive_factors = [
        "F_REV_GROWTH_YOY", "F_EPS_GROWTH_YOY", "F_EBITDA_GROWTH_YOY",
        "F_EPS_QOQ_TREND", "F_ROE", "F_ROIC", "F_OP_MARGIN", "F_NET_MARGIN",
        "F_ASSET_TURNOVER", "F_FCF_PER_SHARE", "F_EARNINGS_YIELD",
        "MOMENTUM_240", "MOMENTUM_80"
    ]

    negative_factors = [
        "F_NET_DEBT_EBITDA", "F_CASH_CONV_CYCLE",
        "F_P_BV", "F_P_S", "STD_56D"
    ]
    
    key_cols = ["TICKER", "DATE", "CLOSE"]

    df = (
        spark.read.parquet("data/model_data.parquet")
        .select(*key_cols, *positive_factors, *negative_factors)   
    )
    
    if start_date is not None:
        df = df.filter(F.col("DATE") >= start_date)
    
    if end_date is not None:
        df = df.filter(F.col("DATE") >= end_date)
    return df


def bt_rank(df_rank, rank_cutoff=10, rebalance_days=60):

    w_days = Window.partitionBy(F.lit(1)).orderBy("DATE")

    days_with_reb_period = df_rank.select("DATE").distinct() \
        .withColumn("REB_PERIOD", F.floor((F.row_number().over(w_days) - 1) / F.lit(rebalance_days)))

    days_reb_start = days_with_reb_period.groupBy("REB_PERIOD").agg(F.min("DATE").alias("REB_DATE"))

    # THIS SHOULD PROVIDE TICKERS VALID FOR GIVEN REB_PERIOD
    tickers_for_reb_periods = df_rank \
        .filter(F.col("SCORE_RANK") <= rank_cutoff) \
        .select("TICKER", "DATE") \
        .join(days_reb_start.withColumnRenamed("REB_DATE", "DATE"), on="DATE", how="inner") \
        .drop("DATE").orderBy("REB_PERIOD")

    port_w = Window.partitionBy("REB_PERIOD").orderBy("DATE")
    port_wt = Window.partitionBy("REB_PERIOD", "TICKER").orderBy("DATE")

    # Now this contains tickers selected in rebalancing only over time
    rebalanced_df = df_rank \
        .join(days_with_reb_period, on="DATE", how="inner") \
        .join(tickers_for_reb_periods, on=["TICKER", "REB_PERIOD"], how='inner') \
        .withColumn("RET", F.coalesce(F.col("CLOSE") / F.lag("CLOSE", 1).over(port_wt), F.lit(1))) \
        .orderBy("REB_PERIOD", "DATE", "TICKER")
    
    
    # calc avg price for portfolio
    portfolio_df = rebalanced_df \
        .groupBy("REB_PERIOD", "DATE").agg(F.avg("RET").alias("PORTFOLIO_AVG_PRICE")) \
        .withColumn("LOG_RET", F.log(F.col("PORTFOLIO_AVG_PRICE"))) \
        .withColumn("CUMSUM_LOG_RET", F.sum(F.col("LOG_RET")).over(Window.partitionBy(F.lit(1)).orderBy("DATE").rowsBetween(Window.unboundedPreceding, 0))) \
        .withColumn("PORTFOLIO_VALUE", F.exp(F.col("CUMSUM_LOG_RET"))) \
        .orderBy("REB_PERIOD", "DATE")
    
    return portfolio_df.select("DATE", "PORTFOLIO_VALUE")


def plot_portfolio_value(portfolio_dfs, labels):
    import matplotlib.pyplot as plt
    import pandas as pd
    from tqdm import tqdm

    plt.figure(figsize=(14, 6))

    # Loop through each portfolio DataFrame
    for df, label in tqdm(zip(portfolio_dfs, labels)):

        # Convert Spark → Pandas
        pdf = df.orderBy("DATE").toPandas()

        # Ensure DATE is datetime
        pdf["DATE"] = pd.to_datetime(pdf["DATE"])

        # Plot portfolio value
        plt.plot(pdf["DATE"], pdf["PORTFOLIO_VALUE"], label=label)

    # Labels and formatting
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Performance Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save to PNG
    plt.savefig("portfolio_bt.png", dpi=300)
    plt.close()


def save_portfolio_plot(df_portfolio, filename="portfolio_performance.png"):
    """
    Creates and saves a PNG plot of cumulative portfolio returns over time.
    df_portfolio must contain columns: DATE, PORTFOLIO_RETURN
    """

    import matplotlib.pyplot as plt
    import pandas as pd

    # Convert Spark → Pandas
    pdf = df_portfolio.orderBy("DATE").toPandas()

    # Ensure DATE is datetime
    pdf["DATE"] = pd.to_datetime(pdf["DATE"])

    # Compute cumulative return
    pdf["CUM_RETURN"] = (1 + pdf["PORTFOLIO_RETURN"]).cumprod() - 1

    # Plot
    plt.figure(figsize=(14, 6))
    # plt.plot(pdf["DATE"], pdf["PORTFOLIO_RETURN"], label="Cumulative Return", color="blue")
    plt.plot(pdf["DATE"], pdf["CUM_RETURN"], label="Cumulative Return", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Portfolio Performance Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save to PNG
    plt.savefig(filename, dpi=300)
    plt.close()

    return filename

if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("Load_STOOQ_Prices")
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .config("spark.driver.memory", "15g")
        .getOrCreate()
    )

    
    
    weights = {
        # Momentum
        "MOMENTUM_240": 1.5,
        "MOMENTUM_80": 1,

        # Quality
        "F_ROE": 2.0,
        "F_ROIC": 2.0,
        "F_OP_MARGIN": 1.5,
        "F_NET_MARGIN": 1.5,
        "F_ASSET_TURNOVER": 1.2,

        # Value
        "F_EARNINGS_YIELD": 1.5,
        "F_P_BV": 1.2,
        "F_P_S": 1.2,

        # Growth
        "F_REV_GROWTH_YOY": 1.0,
        "F_EPS_GROWTH_YOY": 1.0,
        "F_EBITDA_GROWTH_YOY": 3.0,
        "F_EPS_QOQ_TREND": 1.0,

        # Risk / Leverage (penalties)
        "F_NET_DEBT_EBITDA": 0.5,
        "F_CASH_CONV_CYCLE": 0.5,
        "STD_56D": 0.3,
    }
    
    df = read_scoring_data(start_date="2025-06-01", end_date=None)
        
    df_score = add_factor_score(df, factor_weights=weights)
    
    df_rank = assign_score_ranking(df_score).cache()
    
   
    dfs = []
    labels = []

    for rebalance_period in [10, 20, 40, 60, 80]:
        labels.append(f"RE_PERIOD_{rebalance_period}")
        dfs.append(bt_rank(df_rank, rank_cutoff=15, rebalance_days=rebalance_period))

    plot_portfolio_value(portfolio_dfs=dfs, labels=labels)

