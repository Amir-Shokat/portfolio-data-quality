-- S06 Rights issue lifecycle across two symbols (the highest-risk scenario).
-- RIGHTS_GRANT on the main symbol and RIGHTS_INIT on the rights symbol (record date);
-- trading or RIGHTS_EXERCISE on the rights symbol; RIGHTS_EXPIRE at the end of the
-- subscription period; RIGHTS_CONVERT on the main symbol (activation date).
-- Select both symbols. Expected report: MATCH on both.
SELECT e.customer_id,
       SUM(e.event_type = 'RIGHTS_INIT')     AS init_rows,
       SUM(e.event_type = 'RIGHTS_EXERCISE') AS exercise_rows,
       SUM(e.event_type = 'RIGHTS_EXPIRE')   AS expire_rows,
       SUM(e.event_type = 'RIGHTS_CONVERT')  AS convert_rows,
       SUM(e.event_type = 'SELL' AND s.instrument_class = 'RIGHT') AS rights_sold
FROM position_events e JOIN symbols s ON s.symbol = e.symbol
WHERE e.is_deleted = 0
GROUP BY e.customer_id
HAVING SUM(e.event_type = 'RIGHTS_INIT') > 0
ORDER BY exercise_rows DESC, rights_sold DESC, e.customer_id;
