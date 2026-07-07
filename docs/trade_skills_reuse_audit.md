# Trade Skills Reuse Audit

Source reviewed: `/Users/serena/Documents/Notes/Trade/Github/trade-skills-main`.

The local checkout has no `LICENSE` file and appears to be a personal trading journal/workbench rather than a packaged public library. RunJin therefore does not copy source code wholesale. The integration uses clean-room Python implementations that absorb product ideas, data shapes, and workflow concepts.

## Useful Ideas Absorbed

| Area | Trade Skills idea | RunJin integration |
| --- | --- | --- |
| Symbol cockpit | One stock page with prediction, environment, news, review, note, AI comment | `Symbol Cockpit` Streamlit page |
| SEPA dashboard | Minervini trend template, 50/150/200 MA, 52-week checks, RS vs benchmark | `src/trade_skills/sepa.py` and `SEPA Lab` |
| Intraday signal | 5m/15m/1h direction, anchor price, scenarios, risk gates | `src/trade_skills/intraday.py` and `Intraday Signal` |
| Capital rotation | Cohort flow comparison and narrative labels | `src/trade_skills/capital_rotation.py` and `Capital Rotation` |
| Journal | Durable stock deep-dive markdown fields | `src/trade_skills/journal.py` and `Research Journal` |

## Not Copied

- Fastify/React/Vite app shell.
- Longbridge streaming implementation.
- Claude skill source text as executable prompts.
- SQLite schema and server routes.
- Any source file without a clear license grant.

## RunJin Boundary

All new features are research-only. They do not place orders, do not connect broker credentials, and do not turn AI or indicator output into executable instructions.
