-- S10 Historical versus current ISIN: the depository history holds an older ISIN of a symbol
-- that has since moved market (the first 11 characters differ; linked in isin_lineage).
-- Expected report: MATCH. If the link cannot be followed, every old-ISIN row becomes
-- MISSING_REBUILT (see S12): a fault in ISIN matching.
SELECT d.customer_id, d.symbol, d.date, d.received_isin, s.isin AS current_isin,
       l.new_isin IS NOT NULL AS linked
FROM depository_daily d
JOIN symbols s ON s.symbol = d.symbol
LEFT JOIN isin_lineage l ON substr(l.old_isin, 1, 11) = substr(d.received_isin, 1, 11)
WHERE substr(d.received_isin, 1, 11) <> substr(s.isin, 1, 11)
ORDER BY d.customer_id, d.date;
