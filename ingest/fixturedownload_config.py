"""
Static configuration for the one-time fixturedownload.com full-season
fixture preload. Separate from ingest/config.py deliberately -- that
file's SOURCE_NAME/LEAGUE_CODES are football-data.co.uk-specific and
imported by the twice-weekly automated loop; this is a different
source used by a different, supervised, one-time script.
"""

SOURCE_NAME = "fixturedownload.com"
BASE_URL = "https://fixturedownload.com/download/csv"

# fixturedownload.com slug -> our competition_id (Schema.sql seeds
# competitions 1-4 as Premier League/Championship/League One/League Two).
# Confirmed live via https://fixturedownload.com/index (Football/Soccer
# section) as of Aug 2026 -- re-check this listing if a slug 404s in a
# future season, fixturedownload's naming for lower divisions in
# particular ("efl-league-one-2026") doesn't obviously follow the
# EPL/Championship pattern.
DIVISIONS = {
    "epl-2026":            1,
    "championship-2026":   2,
    "efl-league-one-2026": 3,
    "efl-league-two-2026": 4,
}

# The filename fixturedownload.com's own download button actually
# produces for each division, as observed from a real manual download --
# distinct from the URL slugs above (DIVISIONS), which are what the site
# uses in its *page URLs*, not necessarily what it names the file once
# you click "download CSV". Keyed the same way (slug -> ...) so
# read_local_csv_rows can look one up directly from a slug.
#
# Confirmed pattern as of Aug 2026: "<comp>-<season_start><season_end>.csv",
# with "championship" prefixed "efl-" (but "epl" is not). Re-check this
# if a future season's download filenames don't match -- easier to fix
# here than to ask for the files to be renamed every year.
DOWNLOAD_FILENAMES = {
    "epl-2026":            "epl-20262027.csv",
    "championship-2026":   "efl-championship-20262027.csv",
    "efl-league-one-2026": "efl-league-one-20262027.csv",
    "efl-league-two-2026": "efl-league-two-20262027.csv",
}