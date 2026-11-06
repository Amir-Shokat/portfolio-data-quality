-- S03 Admin (manual) registrations in the depository history.
-- Admin rows cannot be rebuilt from settlements, so the rebuild passes them through and the
-- depository history usually matches. The effect shows in the ledger as an UNEXPLAINED
-- plug, which check A1 reports.
-- Expected report: no difference; the plug is rebuilt the same way.
SELECT d.customer_id, d.symbol, d.date, d.quantity, d.registration_type,
       (SELECT COUNT(*) FROM position_events e
         WHERE e.customer_id = d.customer_id AND e.symbol = d.symbol
           AND e.event_type LIKE 'UNEXPLAINED%') AS plug_rows
FROM depository_daily d
WHERE d.registration_type = 'ADMIN'
ORDER BY d.customer_id, d.symbol, d.date;
