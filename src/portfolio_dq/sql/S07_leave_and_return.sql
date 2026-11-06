-- S07 Customer leaves and returns (custodian change out and back in).
-- Finds: a zero-quantity custodian row directly followed by a JOIN whose quantity equals the
-- quantity before the leave. The rebuild marks the leave row IGNORE.
-- Expected report: MISMATCH on depository_daily.row_type if storage did not mark it.
SELECT l.customer_id, l.symbol, l.date AS leave_date, l.row_type AS stored_row_type,
       j.date AS join_date, j.quantity AS join_quantity
FROM depository_daily l
JOIN depository_daily j
  ON j.customer_id = l.customer_id AND j.symbol = l.symbol AND j.row_type = 'JOIN'
 AND j.date = (SELECT MIN(x.date) FROM depository_daily x
               WHERE x.customer_id = l.customer_id AND x.symbol = l.symbol AND x.date > l.date)
WHERE l.quantity = 0 AND l.row_type IN ('LEAVE', 'IGNORE')
  AND j.quantity = (SELECT y.quantity FROM depository_daily y
                    WHERE y.customer_id = l.customer_id AND y.symbol = l.symbol
                      AND y.date < l.date AND y.row_type <> 'IGNORE'
                    ORDER BY y.date DESC LIMIT 1)
ORDER BY l.customer_id;
