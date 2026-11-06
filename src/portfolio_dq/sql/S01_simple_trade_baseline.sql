-- S01 Simple-trade baseline (negative control). Run it first: a difference here is a fault
-- in the rebuild itself, not in the data.
-- Finds: customer+symbol units with only buys and sells. No corporate action, no transfer,
-- no admin registration, no ignored or change-market depository row, no soft delete.
-- Expected report: no difference in any table.
SELECT e.customer_id, e.symbol, COUNT(*) AS trades, MIN(e.date) AS first_trade
FROM position_events e
GROUP BY e.customer_id, e.symbol
HAVING SUM(e.event_type NOT IN ('BUY', 'SELL')) = 0
   AND SUM(e.is_deleted) = 0
   AND NOT EXISTS (SELECT 1 FROM depository_daily d
                   WHERE d.customer_id = e.customer_id AND d.symbol = e.symbol
                     AND (d.row_type <> 'POSITION' OR d.registration_type <> 'DEPOSITORY'))
ORDER BY e.customer_id, e.symbol;
