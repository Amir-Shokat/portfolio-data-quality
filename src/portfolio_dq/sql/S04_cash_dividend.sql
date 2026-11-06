-- S04 Cash dividend.
-- Finds: customers who held the symbol at the close before the record date, and how many
-- dividend rows they received.
-- The rebuild creates a CASH_DIVIDEND row (quantity 0, ca_value = dividend per share,
-- time 08:15). Expected report: MATCH if present; MISSING_STORED on the ledger row (and on
-- the daily row of that date) if it was never written.
WITH ca AS (
  SELECT a.ca_id, a.symbol,
         (SELECT MAX(c.date) FROM calendar c WHERE c.date < a.record_date) AS day_before
  FROM corporate_actions a WHERE a.action_type = 'DIVIDEND'
), held AS (
  SELECT ca.ca_id, p.customer_id, p.symbol, p.quantity
  FROM ca JOIN positions_daily p ON p.symbol = ca.symbol
  WHERE p.date = (SELECT MAX(q.date) FROM positions_daily q
                  WHERE q.customer_id = p.customer_id AND q.symbol = p.symbol
                    AND q.date <= ca.day_before)
    AND p.quantity > 0
)
SELECT h.customer_id, h.symbol, h.ca_id, h.quantity AS eligible_quantity,
       (SELECT COUNT(*) FROM position_events e
         WHERE e.customer_id = h.customer_id AND e.ca_id = h.ca_id
           AND e.event_type = 'CASH_DIVIDEND' AND e.is_deleted = 0) AS dividend_rows
FROM held h
ORDER BY dividend_rows, h.customer_id;
