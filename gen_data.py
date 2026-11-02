"""Generate synthetic field-force data: roster (attendance) and store visits (coverage + OSA).

Outputs two CSVs next to this script:
  roster.csv  - one row per rep per working day (present 0/1)
  visits.csv  - one row per planned store visit (completed 0/1, SKUs audited / available)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
END = pd.Timestamp("2026-10-28")
DAYS = 30
OUT = Path(__file__).parent

REGIONS = {
    # region: (reps, attendance mean, coverage mean, osa mean, localities)
    "West": (10, 0.945, 0.915, 0.905, ["Andheri", "Thane", "Kothrud", "Navrangpura", "Vashi", "Baner"]),
    "North": (9, 0.925, 0.895, 0.880, ["Saket", "Gurugram Sec 29", "Sector 18 Noida", "Rajouri Garden", "Model Town", "Sushant Lok"]),
    "South": (10, 0.955, 0.930, 0.915, ["Indiranagar", "T Nagar", "Banjara Hills", "Koramangala", "Adyar", "Whitefield"]),
    "East": (6, 0.905, 0.860, 0.845, ["Salt Lake", "Park Street", "Patia", "Dispur", "Howrah", "Gariahat"]),
    "Central": (5, 0.915, 0.880, 0.860, ["Vijay Nagar", "MP Nagar", "Civil Lines", "Sitabuldi", "Ratanpur", "Shyamla Hills"]),
}
CHAINS = ["Daily Basket", "Value Hub", "City Mart", "Fresh Corner", "Shelf Plus"]


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = [d for d in pd.date_range(END - pd.Timedelta(days=DAYS - 1), END) if d.dayofweek != 6]  # Sundays off

    # Stores: 6 per region-locality pair keeps names unique and readable
    stores = []
    for region, (_, _, _, osa_mu, locs) in REGIONS.items():
        for i, loc in enumerate(locs):
            for j in range(2):
                chain = CHAINS[(i + j * 2) % len(CHAINS)]
                stores.append({
                    "store": f"{chain} {loc}",
                    "region": region,
                    # store-level quality: a few chronically weak stores per region
                    "store_osa": np.clip(rng.normal(osa_mu, 0.045), 0.62, 0.99),
                })
    stores = pd.DataFrame(stores).drop_duplicates("store").reset_index(drop=True)

    roster_rows, visit_rows = [], []
    for region, (n_reps, att_mu, cov_mu, _, _) in REGIONS.items():
        region_stores = stores[stores.region == region].reset_index(drop=True)
        for r in range(n_reps):
            rep = f"{region[:1]}{r + 1:02d}"
            rep_bias = rng.normal(0, 0.02)
            # mild upward drift over the month for West/South, downward for East
            drift = {"West": 0.0006, "South": 0.0008, "East": -0.0009}.get(region, 0.0)
            for k, d in enumerate(dates):
                p_att = np.clip(att_mu + rep_bias + drift * (k - len(dates) / 2) + (-0.03 if d.dayofweek == 0 else 0), 0.6, 0.995)
                present = int(rng.random() < p_att)
                roster_rows.append((d, region, rep, present))

                planned = int(rng.integers(6, 9))
                for v in range(planned):
                    st = region_stores.iloc[int(rng.integers(0, len(region_stores)))]
                    p_cov = np.clip(cov_mu / att_mu + drift * (k - len(dates) / 2), 0.7, 0.995)
                    done = int(present and rng.random() < p_cov)
                    audited = int(rng.integers(18, 31)) if done else 0
                    avail = int(rng.binomial(audited, np.clip(st.store_osa + drift * (k - len(dates) / 2), 0.5, 0.995))) if done else 0
                    visit_rows.append((d, region, rep, st.store, 1, done, audited, avail))

    roster = pd.DataFrame(roster_rows, columns=["date", "region", "rep", "present"])
    visits = pd.DataFrame(visit_rows, columns=["date", "region", "rep", "store", "planned", "completed", "skus_audited", "skus_available"])
    roster.to_csv(OUT / "roster.csv", index=False)
    visits.to_csv(OUT / "visits.csv", index=False)
    print(f"roster: {len(roster):,} rows | visits: {len(visits):,} rows | stores: {visits.store.nunique()} | days: {len(dates)}")


if __name__ == "__main__":
    main()
