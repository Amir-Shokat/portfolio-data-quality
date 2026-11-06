-- S12 (additional) Lineage link lost. The S10 evidence query, run after the tester removes
-- the link in a disposable copy. Units whose history used the old ISIN lose those rows from
-- the rebuild.
-- Expected report: MISSING_REBUILT on depository_daily and depository_latest.
SELECT d.customer_id, d.symbol, d.date, d.received_isin, s.isin AS current_isin,
       l.new_isin IS NOT NULL AS linked
FROM depository_daily d
JOIN symbols s ON s.symbol = d.symbol
LEFT JOIN isin_lineage l ON substr(l.old_isin, 1, 11) = substr(d.received_isin, 1, 11)
WHERE substr(d.received_isin, 1, 11) <> substr(s.isin, 1, 11)
ORDER BY d.customer_id, d.date;
