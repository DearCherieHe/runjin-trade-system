# Trading Rules

## Account Separation

- Long-term observation and paper-trading experiments are separate workflows.
- Paper bot losses must not be funded from long-term investment capital.
- V0.1 never places real orders.

## Long-term Desk

- Candidates are scored across narrative space, revenue acceleration, margin quality, cash flow, value-chain position, market disagreement, and valuation tolerance.
- A stock needs explicit catalysts and invalidation rules before it can be treated as a serious research candidate.
- The dashboard is a research system, not a recommendation engine.

## Paper Bot

- No leverage.
- Position size is capped by `configs/risk_rules.yaml`.
- Daily loss and max drawdown limits trigger a stop status.
- Kronos-style forecasts are research context only and do not drive order placement.
