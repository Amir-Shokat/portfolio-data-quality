-- S14 (additional) Informational field only: registration type not set in storage.
-- Expected report: MISMATCH with severity INFO on depository_daily.registration_type. An
-- INFO-only unit counts as matched; it must not be reported as an error.
SELECT customer_id, symbol, date, quantity, registration_type
FROM depository_daily
WHERE registration_type IS NULL
ORDER BY customer_id, date;
