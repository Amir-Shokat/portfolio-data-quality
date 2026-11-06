-- S11 Soft-deleted ledger rows (second negative control).
-- The rebuild ignores is_deleted = 1 rows, and so does the comparison.
-- Expected report: no difference. MISSING_REBUILT here would mean a deleted row leaked into
-- one side of the comparison: a fault in the tool, not drift in the data.
SELECT customer_id, symbol, date, time, event_type, quantity, source_ref
FROM position_events
WHERE is_deleted = 1
ORDER BY customer_id, date;
