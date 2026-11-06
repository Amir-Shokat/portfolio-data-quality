-- S02 Asset mismatch between the two snapshots (the main drift scenario).
-- Finds: units where the depository snapshot and the position snapshot disagree.
-- Compares on symbol, not on the canonical ISIN, so a split symbol key also shows here.
-- Expected report: a MISMATCH on the snapshot quantity of the side that is wrong.
SELECT d.customer_id, d.symbol, d.quantity AS depository_quantity,
       p.quantity AS positions_quantity, d.quantity - COALESCE(p.quantity, 0) AS diff
FROM depository_latest d
LEFT JOIN positions_latest p ON p.customer_id = d.customer_id AND p.symbol = d.symbol
WHERE d.quantity <> COALESCE(p.quantity, 0)
UNION ALL
SELECT p.customer_id, p.symbol, NULL, p.quantity, -p.quantity
FROM positions_latest p
WHERE p.quantity <> 0
  AND NOT EXISTS (SELECT 1 FROM depository_latest d
                  WHERE d.customer_id = p.customer_id AND d.symbol = p.symbol)
ORDER BY 1, 2;
