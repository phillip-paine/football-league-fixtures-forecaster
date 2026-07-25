-- English Soccer Forecast Engine — core schema
-- SQLite, STRICT tables, no ORM. Designed to never need a migration:
-- open-ended data (odds, stats, ratings) is modeled as rows, not columns.
--
-- Every connection must run this once (SQLite defaults it off):
--   PRAGMA foreign_keys = ON;

-- ============================================================
-- DIMENSION TABLES
-- ============================================================

CREATE TABLE teams (
    team_id        INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    country        TEXT NOT NULL DEFAULT 'England'
) STRICT;

-- Every source spells team names differently ("Man United" vs
-- "Manchester Utd" vs "Manchester United"). Resolve through this
-- table instead of in ingestion code: a new source or a new spelling
-- is an INSERT, never a schema change.
CREATE TABLE team_aliases (
    team_id    INTEGER NOT NULL REFERENCES teams(team_id),
    source     TEXT NOT NULL,   -- 'football-data.co.uk','clubelo','fbref','api-football','football-data.org'
    alias_name TEXT NOT NULL,
    PRIMARY KEY (source, alias_name)
) STRICT;

CREATE INDEX idx_team_aliases_team ON team_aliases(team_id);

CREATE TABLE competitions (
    competition_id INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,   -- 'Premier League', 'Championship', ...
    country        TEXT NOT NULL DEFAULT 'England',
    tier           INTEGER NOT NULL        -- 1 = top flight; feeds the model's
                                            -- division-level pooling directly
) STRICT;

CREATE TABLE competition_aliases (
    competition_id INTEGER NOT NULL REFERENCES competitions(competition_id),
    source         TEXT NOT NULL,
    alias_code     TEXT NOT NULL,   -- football-data.co.uk codes: 'E0','E1','E2','E3'
    PRIMARY KEY (source, alias_code)
) STRICT;

CREATE TABLE seasons (
    season_id  INTEGER PRIMARY KEY,
    start_year INTEGER NOT NULL UNIQUE,  -- 2025 -> the 2025-26 season
    label      TEXT NOT NULL             -- '2025-2026'
) STRICT;

-- Which competition (and therefore tier) a team played in for a given
-- season. This is the promoted-team problem made explicit: division
-- membership is a fact about a season, not a fixed property of a
-- team. The model's team -> division -> competition hierarchy reads
-- straight off this table.
CREATE TABLE team_season (
    team_id        INTEGER NOT NULL REFERENCES teams(team_id),
    season_id      INTEGER NOT NULL REFERENCES seasons(season_id),
    competition_id INTEGER NOT NULL REFERENCES competitions(competition_id),
    PRIMARY KEY (team_id, season_id)
) STRICT;

CREATE TABLE players (
    player_id      INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL
) STRICT;

CREATE TABLE player_aliases (
    player_id  INTEGER NOT NULL REFERENCES players(player_id),
    source     TEXT NOT NULL,
    alias_name TEXT NOT NULL,
    PRIMARY KEY (source, alias_name)
) STRICT;

-- ============================================================
-- CROSS-SOURCE IDENTITY MAP
-- ============================================================

-- football-data.co.uk, football-data.org, API-Football and clubelo
-- each use their own IDs for the same match/team/player. Rather than
-- adding a nullable *_id column to core tables per source (a
-- migration every time a source is onboarded), new sources just add
-- rows here. Also makes twice-weekly re-ingestion idempotent: check
-- external_id before inserting.
CREATE TABLE external_ids (
    entity_type TEXT NOT NULL CHECK (entity_type IN ('match','team','player')),
    entity_id   INTEGER NOT NULL,
    source      TEXT NOT NULL,
    external_id TEXT NOT NULL,
    PRIMARY KEY (source, entity_type, external_id)
) STRICT;

CREATE INDEX idx_external_ids_entity ON external_ids(entity_type, entity_id);

-- ============================================================
-- CORE FACT TABLE
-- ============================================================

CREATE TABLE matches (
    match_id       INTEGER PRIMARY KEY,
    season_id      INTEGER NOT NULL REFERENCES seasons(season_id),
    competition_id INTEGER NOT NULL REFERENCES competitions(competition_id),
    match_date     TEXT NOT NULL,     -- ISO-8601 'YYYY-MM-DD'
    kickoff_time   TEXT,              -- ISO-8601 'HH:MM'; nullable, not always known historically
    home_team_id   INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id   INTEGER NOT NULL REFERENCES teams(team_id),
    home_goals     INTEGER,           -- NULL until played
    away_goals     INTEGER,
    home_goals_ht  INTEGER,
    away_goals_ht  INTEGER,
    status         TEXT NOT NULL DEFAULT 'scheduled'
                     CHECK (status IN ('scheduled','played','postponed','abandoned')),
    referee        TEXT,
    attendance     INTEGER,           -- fixed, single-valued per-match fact; not per-team, so it
                                        -- doesn't belong in the match_team_stats EAV table below
    CHECK (home_team_id != away_team_id)
) STRICT;

CREATE INDEX idx_matches_date        ON matches(match_date);
CREATE INDEX idx_matches_home        ON matches(home_team_id);
CREATE INDEX idx_matches_away        ON matches(away_team_id);
CREATE INDEX idx_matches_season_comp ON matches(season_id, competition_id);

-- ============================================================
-- OPEN-ENDED SATELLITE DATA (EAV-style — this is the part that
-- absorbs change without ever touching the schema)
-- ============================================================

-- Odds columns in football-data.co.uk grow/shrink almost every season
-- (Bet365, Pinnacle, William Hill, market average, market max, ...).
-- One row per (match, bookmaker, outcome) means a new bookmaker is an
-- INSERT, not an ALTER TABLE.
CREATE TABLE match_odds (
    match_id  INTEGER NOT NULL REFERENCES matches(match_id),
    bookmaker TEXT NOT NULL,          -- 'bet365','pinnacle','market_avg','market_max', ...
    outcome   TEXT NOT NULL CHECK (outcome IN ('H','D','A')),
    odds      REAL NOT NULL,
    PRIMARY KEY (match_id, bookmaker, outcome)
) STRICT;

-- Same logic for match stats: shots, shots-on-target, corners, cards,
-- fouls (football-data.co.uk), xG/xGA (Understat/FBref, 2014-17+),
-- and whatever the next source adds.
CREATE TABLE match_team_stats (
    match_id   INTEGER NOT NULL REFERENCES matches(match_id),
    team_id    INTEGER NOT NULL REFERENCES teams(team_id),
    stat_name  TEXT NOT NULL,     -- 'shots','shots_on_target','corners','fouls',
                                   -- 'yellow_cards','red_cards','xg','xga','possession_pct', ...
    stat_value REAL NOT NULL,
    PRIMARY KEY (match_id, team_id, stat_name)
) STRICT;

-- Time-varying team strength from any source: clubelo now, your own
-- model's posterior attack/defense strength later, a Phase-2
-- state-space rating after that. A new rating series is a new
-- rating_type value, never a new column or table.
CREATE TABLE team_ratings (
    team_id      INTEGER NOT NULL REFERENCES teams(team_id),
    as_of_date   TEXT NOT NULL,     -- ISO-8601
    rating_type  TEXT NOT NULL,     -- 'clubelo','model_attack','model_defense','model_home_adv', ...
    rating_value REAL NOT NULL,
    PRIMARY KEY (team_id, as_of_date, rating_type)
) STRICT;

CREATE INDEX idx_ratings_type_date ON team_ratings(rating_type, as_of_date);

-- ============================================================
-- FIXED-SHAPE SATELLITE DATA (genuinely scoped feature sets —
-- plain columns are the right call here, not EAV)
-- ============================================================

-- Weather is precisely scoped already (precip + wind at kickoff,
-- one source: Open-Meteo), so plain columns beat EAV overhead here.
CREATE TABLE match_weather (
    match_id    INTEGER PRIMARY KEY REFERENCES matches(match_id),
    temp_c      REAL,
    precip_mm   REAL,
    wind_kph    REAL,
    is_forecast INTEGER NOT NULL DEFAULT 0  -- 0 = observed after the fact, 1 = forecast at ingestion time
) STRICT;

-- Full lineups: backfills the historical key-player feature and
-- captures confirmed lineups (~1hr pre-kickoff via API-Football).
CREATE TABLE match_lineups (
    match_id    INTEGER NOT NULL REFERENCES matches(match_id),
    team_id     INTEGER NOT NULL REFERENCES teams(team_id),
    player_id   INTEGER NOT NULL REFERENCES players(player_id),
    is_starting INTEGER NOT NULL,          -- 0/1
    is_captain  INTEGER NOT NULL DEFAULT 0,
    position    TEXT,
    PRIMARY KEY (match_id, player_id)
) STRICT;

CREATE INDEX idx_lineups_match_team ON match_lineups(match_id, team_id);

-- Simplified key-player availability (GK + captain + top scorer only,
-- per design notes). Kept as its own narrow table rather than derived
-- on the fly: "top scorer" needs a point-in-time computation (leading
-- scorer as of that gameweek) that's worth freezing once computed.
CREATE TABLE key_player_status (
    match_id  INTEGER NOT NULL REFERENCES matches(match_id),
    team_id   INTEGER NOT NULL REFERENCES teams(team_id),
    role      TEXT NOT NULL CHECK (role IN ('gk','captain','top_scorer')),
    player_id INTEGER REFERENCES players(player_id),
    available INTEGER NOT NULL,   -- 0/1
    source    TEXT NOT NULL CHECK (source IN ('lineup_confirmed','injury_predicted')),
    PRIMARY KEY (match_id, team_id, role)
) STRICT;

-- ============================================================
-- OPERATIONAL
-- ============================================================

-- Debugging trail for the weekly/twice-weekly ingestion loop.
CREATE TABLE ingestion_runs (
    run_id       INTEGER PRIMARY KEY,
    source       TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    rows_written INTEGER,
    status       TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','success','failed')),
    notes        TEXT
) STRICT;

-- ============================================================
-- REFERENCE DATA (stable, known in advance — safe to seed now)
-- ============================================================

INSERT INTO competitions (competition_id, name, country, tier) VALUES
    (1, 'Premier League', 'England', 1),
    (2, 'Championship',   'England', 2),
    (3, 'League One',     'England', 3),
    (4, 'League Two',     'England', 4);

INSERT INTO competition_aliases (competition_id, source, alias_code) VALUES
    (1, 'football-data.co.uk', 'E0'),
    (2, 'football-data.co.uk', 'E1'),
    (3, 'football-data.co.uk', 'E2'),
    (4, 'football-data.co.uk', 'E3');