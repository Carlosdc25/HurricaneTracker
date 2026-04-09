import pandas as pd
from pathlib import Path

class StormDatabase:
    COLUMNS = [
        "ID", "Name", "Date", "Time", "Event", "Status",
        "Latitude", "Longitude", "Maximum Wind", "Minimum Pressure",
        "Low Wind NE", "Low Wind SE", "Low Wind SW", "Low Wind NW",
        "Moderate Wind NE", "Moderate Wind SE", "Moderate Wind SW", "Moderate Wind NW",
        "High Wind NE", "High Wind SE", "High Wind SW", "High Wind NW"
    ]

    def __init__(self):
        self.csv_paths = ["data/csv/atlantic.csv", "data/csv/pacific.csv"]
        self.dataFrame = self.load_csvs(self.csv_paths)
        self._clean_data()

    def load_csvs(self, paths):
        frames = []
        for path in paths:
            path = Path(path)

            if not path.exists():
                raise FileNotFoundError(f"Could not find file: {path}")

            dataFrame = pd.read_csv(
                path,
                header=None,
                names=self.COLUMNS,
                skiprows=1, # Skipping first line with ID, Name, Date, Time
                dtype=str
            )

            frames.append(dataFrame)

        if not frames:
            raise ValueError("Likely wrong CSV path")

        return pd.concat(frames, ignore_index = True)

    def _clean_data(self) -> None:
        # Taking out white space
        self.dataFrame = self.dataFrame.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

        # Use date/time to create timestamp column
        self.dataFrame["Date"] = self.dataFrame["Date"].astype(str).str.strip()
        self.dataFrame["Time"] = self.dataFrame["Time"].astype(str).str.strip().str.zfill(4)

        self.dataFrame["Timestamp"] = pd.to_datetime(
            self.dataFrame["Date"] + self.dataFrame["Time"],
            format="%Y%m%d%H%M",
            errors="coerce"
        )

        # Convert latitude/longitude to signed floats
        self.dataFrame["Latitude"] = self.dataFrame["Latitude"].apply(self.process_latitude)
        self.dataFrame["Longitude"] = self.dataFrame["Longitude"].apply(self.process_longitude)

        # Convert numeric columns
        numeric_cols = [
            "Maximum Wind", "Minimum Pressure",
            "Low Wind NE", "Low Wind SE", "Low Wind SW", "Low Wind NW",
            "Moderate Wind NE", "Moderate Wind SE", "Moderate Wind SW", "Moderate Wind NW",
            "High Wind NE", "High Wind SE", "High Wind SW", "High Wind NW",
        ]

        for col in numeric_cols:
            self.dataFrame[col] = pd.to_numeric(self.dataFrame[col], errors="coerce")

        # Replace NOAA missing values like -999 with NaN
        self.dataFrame = self.dataFrame.replace(-999, pd.NA)

    # Longitude and Latitude preprocessing functions
    def process_latitude(self, value):
        num = float(value[:-1])
        direction = str(value[-1].upper())

        if direction == "S":
            num *= -1

        return num

    def process_longitude(self, value):
        num = float(value[:-1])
        direction = str(value[-1].upper())

        if direction == "W":
            num *= -1

        return num        

    def getStormIds(self):
        return sorted(self.dataFrame["ID"].unique().tolist())

    def stormCount(self):
        return self.dataFrame["ID"].nunique()

    def totalEntries(self):
        return len(self.dataFrame)

    def getStormRecord(self, storm_id):
        return self.dataFrame[self.dataFrame["ID"] == storm_id].sort_values("Timestamp")
        
    

