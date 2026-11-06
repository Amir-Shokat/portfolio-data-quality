-- S09 Ledger rows that were never calculated (PENDING or FAILED).
-- The rebuild ignores the calculation status and calculates everything.
-- Expected report: MISSING_STORED on positions_daily for the affected day, MISMATCH on the
-- position snapshot and on the ledger balance. PENDING is queue lag; FAILED is a real error.
SELECT customer_id, symbol, date, time, event_type, quantity, calc_status
FROM position_events
WHERE is_deleted = 0 AND calc_status <> 'CALCULATED'
ORDER BY date, customer_id;
