-- S08 Same-day market or ISIN change: the raw feed has two ISINs of one security on one day.
-- The rebuild keeps the row with the larger quantity and marks it CHANGE_MARKET.
-- Expected report: MATCH if storage did the same; otherwise MISMATCH on row_type, or
-- MISSING_REBUILT for a row the rebuild dropped.
SELECT f.customer_id, substr(f.isin, 1, 11) AS canonical_isin, f.date,
       COUNT(DISTINCT f.isin) AS isins, MAX(f.quantity) AS kept_quantity
FROM depository_feed f
GROUP BY f.customer_id, substr(f.isin, 1, 11), f.date
HAVING COUNT(DISTINCT f.isin) > 1
ORDER BY f.customer_id;
