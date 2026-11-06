-- S05 Bonus shares with a record date and an activation date.
-- The rebuild creates BONUS_GRANT on the record date with quantity floor(holding x ratio)
-- and BONUS_ACTIVATE on the activation date.
-- Expected report: MATCH. A missing row is MISSING_STORED; a different quantity is MISMATCH.
SELECT e.customer_id, e.symbol, e.ca_id,
       SUM(CASE WHEN e.event_type = 'BONUS_GRANT' THEN e.quantity END)    AS granted,
       SUM(CASE WHEN e.event_type = 'BONUS_ACTIVATE' THEN e.quantity END) AS activated
FROM position_events e
WHERE e.event_type IN ('BONUS_GRANT', 'BONUS_ACTIVATE') AND e.is_deleted = 0
GROUP BY e.customer_id, e.symbol, e.ca_id
ORDER BY e.customer_id;
