def evaluate_risk(backtest_df, risk_rules):
    limits = risk_rules["risk_limits"]
    if backtest_df.empty:
        return "PAUSE", "No backtest data"

    latest_equity = backtest_df["equity"].iloc[-1]
    prior_equity = backtest_df["equity"].iloc[-2] if len(backtest_df) > 1 else latest_equity
    daily_loss = latest_equity / prior_equity - 1
    max_drawdown = backtest_df["drawdown"].min()

    if daily_loss <= -limits["max_daily_loss_pct"]:
        return "STOP", "Daily loss limit breached"
    if max_drawdown <= -limits["max_drawdown_pct"]:
        return "STOP", "Max drawdown limit breached"
    if risk_rules["capital"].get("no_leverage") is not True:
        return "STOP", "No-leverage rule disabled"
    return "CONTINUE", "Risk limits clear"
