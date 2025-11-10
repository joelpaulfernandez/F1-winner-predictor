-- features/fv_preRace.sql
SELECT
  r.raceid,
  rr.drivernumber AS driver_id,
  rr.teamname AS team_id,
  rr.gridposition,
  rr.qualiposition,
  rr.points,
  rr.position AS finish_position,
  rr.dnf,
  rr.winner,
  rr.airtempmean,
  rr.tracktempmean,
  rr.rainprobproxy
FROM 'data/silver/race_results.parquet' rr
JOIN 'data/silver/races.parquet' r USING(raceid);