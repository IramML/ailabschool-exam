import numpy as np
import pandas as pd

class Utils():
    def __init__(self) -> None:
        file_name = "jena_climate_2009_2016.csv"
        csv_path = "in/{}".format(file_name)
        self.df = pd.read_csv(csv_path)
        self.df = self.df[2::3]
        self.date_time = pd.to_datetime(self.df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

        day = 24*60*60
        year = (365.2425)*day

        timestamp_s = self.date_time.map(pd.Timestamp.timestamp)

        self.df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        self.df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        self.df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        self.df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))


        train_split = int(200000 / 3)

        train_df = self.df[:train_split]

        self.train_mean = train_df.mean()
        self.train_std = train_df.std()

    def cos_sin_day(self, day_value):
        day = 24*60*60
        return (np.cos(day_value * (2 * np.pi / day)), np.sin(day_value * (2 * np.pi / day)))

    def cos_sin_year(self, year_value):
        day = 24*60*60
        year = (365.2425)*day
        return (np.cos(year_value * (2 * np.pi / year)), np.sin(year_value * (2 * np.pi / year)))

    def normalized_to_normal_value(self, value, column):
        return (value * self.train_std[column]) + self.train_mean[column]

    def normalize(self, value, column):
        return (value - self.train_mean[column]) / self.train_std[column]

