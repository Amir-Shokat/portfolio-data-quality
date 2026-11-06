-- S13 (additional) One security under two symbol keys (same first 11 ISIN characters).
-- Finds: customers whose rows use more than one symbol of a fragmented security.
-- Expected report: the rebuild resolves the feed to the exact-match symbol, so stored rows
-- under the other key are MISSING_REBUILT and the rebuilt ones are MISSING_STORED.
WITH sym AS (SELECT symbol, substr(isin, 1, 11) AS canonical FROM symbols),
frag AS (SELECT canonical FROM sym GROUP BY canonical HAVING COUNT(*) > 1),
used AS (
  SELECT customer_id, symbol FROM depository_daily
  UNION SELECT customer_id, symbol FROM positions_daily
)
SELECT u.customer_id, s.canonical, GROUP_CONCAT(DISTINCT u.symbol) AS symbols_used
FROM used u JOIN sym s ON s.symbol = u.symbol JOIN frag f ON f.canonical = s.canonical
GROUP BY u.customer_id, s.canonical
HAVING COUNT(DISTINCT u.symbol) > 1;
