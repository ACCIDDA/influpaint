import math
from inspect import isfunction
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from . import season_axis
from .season_axis import SeasonAxis


import numpy as np
import pandas as pd
import xarray as xr

import datetime

from . import helpers as myutils
from ..datasets import read_datasources
from . import converters


def pad_dataframe(df, season_setup):
    # Make sure gt_df and gt_df_final have values (even if NaN) for all dates in the season
    date_range = pd.date_range(start=df.week_enddate.min(), periods=52, freq='W-SAT')
    locations = df.location_code.unique()
    # Create expanded dataframe with all combinations of dates and locations
    expanded_df = pd.DataFrame([(d, l) for d in date_range for l in locations],
                            columns=['week_enddate', 'location_code'])

    # Calculate the season columns
    expanded_df['fluseason'] = expanded_df.week_enddate.apply(season_setup.get_fluseason_year)
    expanded_df['fluseason_fraction'] = expanded_df.week_enddate.apply(season_setup.get_fluseason_fraction)
    expanded_df['season_week'] = expanded_df.week_enddate.apply(season_setup.get_season_week)

    # Merge with original data to get values where they exist
    padded_df = expanded_df.merge(
        df[['week_enddate', 'location_code', 'value']], 
        on=['week_enddate', 'location_code'], 
        how='left'
    )
    return padded_df


class GroundTruth():
    def __init__(
        self,
        season_first_year: str,
        gt_df: pd.DataFrame,
        gt_df_final: pd.DataFrame,
        mask_date: datetime.datetime,
        season_setup: SeasonAxis,
        channels=1,
        image_size=64,
        previous_data=None,
        dataset_coords: xr.core.coordinates.DataArrayCoordinates = None,
    ):
        self.season_first_year = season_first_year
        self.mask_date = pd.to_datetime(mask_date)
        self.channels = channels
        self.image_size = image_size
        self.season_setup = season_setup

        self.gt_df = gt_df
        self.gt_df_final = gt_df_final

        if previous_data is None:
            previous_data = []
        if isinstance(previous_data, list):
            previous_data = pd.concat(previous_data, ignore_index=True) if previous_data else pd.DataFrame()
        self.previous_data = previous_data

        if dataset_coords is not None:
            # change the flusetup locations to be in the same order as flu_payload_array.coords["place"]
            self.season_setup.reorder_locations(list(dataset_coords["place"].values))

        self.gt_df = self.gt_df[self.gt_df["location_code"].isin(self.season_setup.locations)]
        self.gt_df_final = self.gt_df_final[self.gt_df_final["location_code"].isin(self.season_setup.locations)]

        if not self.previous_data.empty:
            self.previous_data = self.season_setup.add_season_columns(self.previous_data, do_fluseason_year=True)

        self.gt_df = pad_dataframe(self.gt_df, self.season_setup)
        self.gt_df_final = pad_dataframe(self.gt_df_final, self.season_setup)

        last_non_nan_datadate = self.gt_df.week_enddate[self.gt_df.value.notna()].max().to_pydatetime()
        # If the last data_point is not in the last week, we need to update the mask to be in the week after the last data point
        if self.mask_date > last_non_nan_datadate + datetime.timedelta(days=7):
            self.mask_date = last_non_nan_datadate + datetime.timedelta(days=2)
            print(f" WARNING: mask_date is after last non-NaN data date, setting mask_date to {self.mask_date}")

        self.gt_xarr = converters.dataframe_to_xarray(
            self.gt_df,
            season_setup=self.season_setup,
            xarray_name="gt_flusight_incidHosp",
            xarrax_features="incidHosp",
            pad_to=image_size,
        )

        self.gt_final_xarr = converters.dataframe_to_xarray(
            self.gt_df_final,
            season_setup=self.season_setup,
            xarray_name="gt_flusight_incidHos_final",
            xarrax_features="incidHosp",
            pad_to=image_size,
        )

        # Find the largest index of the data dates that are before the mask date
        dates = pd.to_datetime(self.gt_xarr.coords["date"].values)
        self.inpaintfrom_idx = sum(dates < self.mask_date)

        self.gt_keep_mask = np.ones((channels, image_size, image_size))
        self.gt_keep_mask[:, self.inpaintfrom_idx :, :] = 0

        print(f"Masking, >> {self.inpaintfrom_idx} weeks already in data, inpainting the next ones")

    @staticmethod
    def _git_checkout_repo_rev(repo_path, target_date=None, main_branch="main"):
        import pygit2

        repo = pygit2.Repository(repo_path)

        if target_date is not None:
            closest_commit = None
            for commit in repo.walk(repo.head.target, pygit2.GIT_SORT_TIME):
                if commit.commit_time <= target_date.timestamp():
                    closest_commit = commit
                    break

            if closest_commit:
                repo.checkout_tree(closest_commit.tree)
                repo.set_head(closest_commit.id)
                print(
                    f"Checked out commit on {target_date} (SHA: {closest_commit.id}, {commit.commit_time}) for repo {repo_path}"
                )
            else:
                print(f"ERROR: No commit found for the specified date on repo {repo_path}.")
        else:
            repo.checkout("refs/heads/" + main_branch)
            print(f"Restored git repo {repo_path}")

    @staticmethod
    def _flusight_repo_info(season_first_year: str):
        if season_first_year == "2023":
            return "Flusight/2023-2024/FluSight-forecast-hub-official/", "main"
        if season_first_year == "2022":
            return "Flusight/2022-2023/FluSight-forecast-hub-official/", "master"
        if season_first_year == "2024":
            return "Flusight/2024-2025/FluSight-forecast-hub-official/", "main"
        if season_first_year == "2025":
            return "Flusight/2024-2025/FluSight-forecast-hub-official/", "main"
        raise ValueError(f"Unsupported FluSight season_first_year: {season_first_year}")

    @classmethod
    def for_flusight(
        cls,
        season_first_year: str,
        data_date: datetime.datetime,
        mask_date: datetime.datetime,
        from_final_data: bool = False,
        channels=1,
        image_size=64,
        nogit=False,
        payload=None,
        payload_season_first_year=None,
        dataset_coords: xr.core.coordinates.DataArrayCoordinates = None,
    ):
        data_date = pd.to_datetime(data_date)
        repo_path, main_branch = cls._flusight_repo_info(season_first_year)
        if not nogit:
            cls._git_checkout_repo_rev(repo_path, target_date=None, main_branch=main_branch)

        season_setup = SeasonAxis.for_flusight(remove_territories=True, remove_us=True)
        flusight = read_datasources.get_from_epidata(
            dataset=f"flusight{season_first_year}", season_setup=season_setup, write=False
        )
        flusight = season_setup.add_season_columns(flusight, do_fluseason_year=True)
        gt_df_final = flusight[flusight["fluseason"] == int(season_first_year)]

        if from_final_data:
            gt_df = gt_df_final.copy()
        else:
            if not nogit:
                cls._git_checkout_repo_rev(repo_path, target_date=data_date, main_branch=main_branch)
            flusight = read_datasources.get_from_epidata(
                dataset=f"flusight{season_first_year}", season_setup=season_setup, write=False
            )
            flusight = season_setup.add_season_columns(flusight, do_fluseason_year=True)
            gt_df = flusight[flusight["fluseason"] == int(season_first_year)]
            if not nogit:
                cls._git_checkout_repo_rev(repo_path, target_date=None, main_branch=main_branch)

        previous_data = []
        for past_year in [int(season_first_year) - 1, int(season_first_year) - 2]:
            try:
                past_df = read_datasources.get_from_epidata(
                    dataset=f"flusight{past_year}", season_setup=season_setup, write=False
                )
                previous_data.append(past_df)
            except Exception as exc:
                print(f" WARNING: could not load flusight{past_year} for historical data: {exc}")

        if payload is not None:
            if payload_season_first_year is None:
                payload_season_first_year = season_first_year
            payload = season_setup.add_season_columns(payload, do_fluseason_year=True)
            this_payload = payload[payload["fluseason"] == int(payload_season_first_year)]
            gt_df = pd.concat([gt_df, this_payload], ignore_index=True)
            gt_df_final = pd.concat([gt_df_final, this_payload], ignore_index=True)
            previous_data.append(payload)

            location_codes = gt_df.location_code.unique()
            new_locations = pd.DataFrame({"location_code": sorted(location_codes)})
            new_locations["location_code"] = new_locations["location_code"].astype(str)
            new_locations = new_locations.merge(
                season_setup.locations_df, on="location_code", how="left"
            )
            new_locations["location_name"] = new_locations["location_name"].fillna(
                new_locations["location_code"]
            )
            new_locations = new_locations[["location_code", "location_name"]]
            season_setup.update_locations(new_locations)

        return cls(
            season_first_year=season_first_year,
            gt_df=gt_df,
            gt_df_final=gt_df_final,
            mask_date=mask_date,
            season_setup=season_setup,
            channels=channels,
            image_size=image_size,
            previous_data=previous_data,
            dataset_coords=dataset_coords,
        )

    @classmethod
    def from_metrocast(
        cls,
        season_first_year: str,
        data_date: datetime.datetime,
        mask_date: datetime.datetime,
        channels=1,
        image_size=128,
        nogit=False,
        dataset_coords: xr.core.coordinates.DataArrayCoordinates = None,
        repo_path="Flusight/metrocast/flu-metrocast",
        data_path="Flusight/metrocast/flu-metrocast/target-data/latest-data.csv",
        main_branch="main",
    ):
        data_date = pd.to_datetime(data_date)
        if not nogit:
            cls._git_checkout_repo_rev(repo_path, target_date=None, main_branch=main_branch)

        season_setup = SeasonAxis.for_metrocast()

        latest_df = pd.read_csv(data_path, parse_dates=["target_end_date"])
        latest_df = latest_df.rename(
            columns={
                "target_end_date": "week_enddate",
                "location": "location_code",
                "observation": "value",
            }
        )
        latest_df["location_code"] = latest_df["location_code"].astype(str).str.strip()
        latest_df["target"] = latest_df["target"].astype(str).str.strip()

        flu_df = latest_df[latest_df["target"] == "Flu ED visits pct"].copy()
        ili_df = latest_df[latest_df["target"] == "ILI ED visits pct"].copy()

        # Combine both targets: use Flu ED visits pct when available, fill gaps with ILI ED visits pct.
        flu_key = flu_df[["week_enddate", "location_code"]].drop_duplicates()
        ili_fill = ili_df.merge(flu_key, on=["week_enddate", "location_code"], how="left", indicator=True)
        ili_fill = ili_fill[ili_fill["_merge"] == "left_only"].drop(columns="_merge")

        flu_df["target_source"] = "Flu ED visits pct"
        ili_fill["target_source"] = "ILI ED visits pct"
        latest_df = pd.concat([flu_df, ili_fill], ignore_index=True)

        full_df = season_setup.add_season_columns(latest_df, do_fluseason_year=True)
        gt_df_final = full_df[full_df["fluseason"] == int(season_first_year)]

        if not nogit:
            cls._git_checkout_repo_rev(repo_path, target_date=data_date, main_branch=main_branch)
            latest_df = pd.read_csv(data_path, parse_dates=["target_end_date"])
            latest_df = latest_df.rename(
                columns={
                    "target_end_date": "week_enddate",
                    "location": "location_code",
                    "observation": "value",
                }
            )
            latest_df["location_code"] = latest_df["location_code"].astype(str).str.strip()
            latest_df["target"] = latest_df["target"].astype(str).str.strip()

            flu_df = latest_df[latest_df["target"] == "Flu ED visits pct"].copy()
            ili_df = latest_df[latest_df["target"] == "ILI ED visits pct"].copy()

            flu_key = flu_df[["week_enddate", "location_code"]].drop_duplicates()
            ili_fill = ili_df.merge(flu_key, on=["week_enddate", "location_code"], how="left", indicator=True)
            ili_fill = ili_fill[ili_fill["_merge"] == "left_only"].drop(columns="_merge")

            flu_df["target_source"] = "Flu ED visits pct"
            ili_fill["target_source"] = "ILI ED visits pct"
            latest_df = pd.concat([flu_df, ili_fill], ignore_index=True)
            cls._git_checkout_repo_rev(repo_path, target_date=None, main_branch=main_branch)
        full_df = season_setup.add_season_columns(latest_df, do_fluseason_year=True)
        gt_df = full_df[full_df["fluseason"] == int(season_first_year)]

        previous_data = full_df.copy()

        return cls(
            season_first_year=season_first_year,
            gt_df=gt_df,
            gt_df_final=gt_df_final,
            mask_date=mask_date,
            season_setup=season_setup,
            channels=channels,
            image_size=image_size,
            previous_data=previous_data,
            dataset_coords=dataset_coords,
        )

    def plot(self):
        season_start_date = datetime.date(int(self.season_first_year), self.season_setup.season_start_month, self.season_setup.season_start_day)
        n_locations = len(self.season_setup.locations)
        n_plots = n_locations + 1  # include US aggregate
        n_cols = math.ceil(math.sqrt(n_plots))
        n_rows = math.ceil(n_plots / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=True,
            figsize=(max(8, n_cols * 2.2), max(8, n_rows * 2.0)),
        )
        gt_piv  = self.gt_df.pivot(index = "week_enddate", columns='location_code', values='value')
        gt_piv_final = self.gt_df_final.pivot(index = "week_enddate", columns='location_code', values='value')
        axes_flat = np.atleast_1d(axes).flat
        ax = axes_flat[0]
        ax.plot(gt_piv[self.season_setup.locations].sum(axis=1), color="black", linewidth=2,label="datadate")
        ax.plot(gt_piv_final[self.season_setup.locations].sum(axis=1), lw=1, color='r', ls='-.', label="final")
        ax.legend()
        ax.set_ylim(0)
        ax.set_title("US")
        for idx, pl in enumerate(gt_piv.columns):
            if idx + 1 >= n_plots:
                break
            ax = axes_flat[idx + 1]
            ax.plot(gt_piv[pl], lw=2, color='k')
            ax.plot(gt_piv_final[pl], lw=1, color='r', ls='-.')
            na_mask = gt_piv.isna()
            ax.plot(gt_piv[na_mask].index,
                    gt_piv[na_mask],
                    marker='o', 
                    color="pink",
                    fillstyle='full', 
                    markeredgecolor='red', 
                    markersize=5,
                    markeredgewidth=1)
            ax.set_title(self.season_setup.get_location_name(pl))
            #ax.grid()
            ax.set_ylim(0)
            ax.set_xlim(season_start_date, season_start_date + datetime.timedelta(days=365))
            #ax.set_xticks(season_setup.get_dates(52).resample("M"))
            #ax.plot(pd.date_range(season_setup.fluseason_startdate, season_setup.fluseason_startdate + datetime.timedelta(days=64*7), freq="W-SAT"), data.flu_dyn[-50:,0,:,idx].T, c='r', lw=.5, alpha=.2)
        for extra_ax in list(axes_flat)[n_plots:]:
            extra_ax.set_visible(False)
        fig.tight_layout()
        fig.autofmt_xdate()

    def _get_historical_series(self, location_code):
        if self.previous_data is None or self.previous_data.empty:
            return []

        if "season_week" not in self.previous_data.columns:
            self.previous_data = self.season_setup.add_season_columns(self.previous_data, do_fluseason_year=True)

        calendar = self.season_setup.get_season_calendar(int(self.season_first_year))
        calendar = calendar[["season_week", "saturday"]]

        hist = self.previous_data[self.previous_data["location_code"] == location_code].copy()
        if hist.empty:
            return []

        series = []
        for hist_season in sorted(hist["fluseason"].dropna().unique()):
            if int(hist_season) == int(self.season_first_year):
                continue
            season_data = hist[hist["fluseason"] == hist_season][["season_week", "value"]].dropna()
            if season_data.empty:
                continue
            season_data = season_data.groupby("season_week", as_index=False)["value"].mean()
            season_data = season_data.merge(calendar, on="season_week", how="inner").sort_values("season_week")
            if season_data.empty:
                continue
            series.append(
                (
                    hist_season,
                    pd.to_datetime(season_data["saturday"]).to_numpy(),
                    season_data["value"].to_numpy(),
                )
            )
        return series

    def plot_mask(self):
        # check that it stitch
        fig, axes = plt.subplots(1, 4, figsize=(8,8), dpi=200, sharex=True, sharey=True)
        import matplotlib as mpl
        cmap_greys = mpl.colormaps.get_cmap('Greys')
        cmap_rainbow = mpl.colormaps.get_cmap("rainbow")
        cmap_greys.set_bad(color='red')
        cmap_rainbow.set_bad(color='red')
        axes[0].imshow(self.gt_xarr.data[0], cmap=cmap_greys)
        axes[0].set_title("Current data rev", fontsize=8)

        axes[1].imshow(self.gt_keep_mask[0], alpha=.3, cmap = cmap_rainbow)
        axes[1].set_title("Inpainting mask", fontsize=8)
        


        axes[2].imshow(self.gt_xarr.data[0], cmap=cmap_greys)
        axes[2].imshow(self.gt_keep_mask[0], alpha=.3, cmap = cmap_rainbow)
        axes[3].set_title("Current data rev", fontsize=8)

        axes[3].imshow(self.gt_final_xarr.data[0], cmap=cmap_greys)
        axes[3].imshow(self.gt_keep_mask[0], alpha=.3, cmap = cmap_rainbow)
        axes[3].set_title("Final data", fontsize=8)

    def export_forecasts(self, fluforecasts_ti, forecasts_national, directory=".", prefix="", forecast_date=None, save_plot=True, nochecks=False):
        forecast_date_str=str(forecast_date)
        if forecast_date == None:
            forecast_date = self.mask_date

        # Calculate season start date for date range calculations
        season_start_date = datetime.date(int(self.season_first_year), self.season_setup.season_start_month, self.season_setup.season_start_day)

        target_dates = pd.date_range(forecast_date, forecast_date + datetime.timedelta(days=4*7), freq="W-SAT")

        target_dict= dict(zip(
            target_dates, 
            [f"{n} wk ahead inc flu hosp" for n in range(1,5)]))

        print(target_dates)
        #pd.DataFrame(colums=["forecast_date","target_end_date","location","type","quantile","value","target"])
        df_list=[]
        for qt in myutils.flusight_quantiles:
            a =  pd.DataFrame(np.quantile(fluforecasts_ti[:,:,:,:len(self.season_setup.locations)], qt, axis=0)[0], 
                    columns= self.season_setup.locations, index=pd.date_range(season_start_date, season_start_date + datetime.timedelta(days=self.image_size*7), freq="W-SAT")).loc[target_dates]
            #a["US"] = a.sum(axis=1)
            a["US"] = pd.DataFrame(np.quantile(forecasts_national, qt, axis=0)[0],
                    columns= ["US"], index=pd.date_range(season_start_date, season_start_date + datetime.timedelta(days=self.image_size*7), freq="W-SAT")).loc[target_dates]

            a = a.reset_index().rename(columns={'index': 'target_end_date'})
            a = pd.melt(a,id_vars="target_end_date",var_name="location")
            a["quantile"] = '{:<.3f}'.format(qt)
            
            df_list.append(a)

        df = pd.concat(df_list)
        df["forecast_date"] = forecast_date_str
        df["type"] = "quantile"
        df["target"] = df["target_end_date"].map(target_dict)
        df = df[["forecast_date","target_end_date","location","type","quantile","value","target"]]
        df

        for col in df.columns:
            print(col)
            print(df[col].unique())

        if not nochecks:
            assert sum(df["value"]<0) == 0
            assert sum(df["value"].isna()) == 0

        # check for Error when validating format: Entries in `value` must be non-decreasing as quantiles increase:
        for tg in target_dates:
            old_vals = np.zeros(len(self.season_setup.locations)+1)
            for dfd in df_list:  # very important to not call this df: it overwrites in namesapce the exported df
                new_vals = dfd[dfd["target_end_date"]==tg]["value"].to_numpy()
                if not (new_vals-old_vals >= 0).all():
                    num_negative = sum((new_vals-old_vals) < 0)
                    print(f" !!!! Quantile validation failed: {num_negative} negative values on {tg}")
                else:
                    pass
                    #print(f"""ok for {dfd["quantile"].unique()}, {tg}""")
                old_vals = new_vals

        df.to_csv(f"{directory}/{prefix}-{forecast_date_str}.csv", index=False)

        if save_plot:
            self.plot_forecasts(fluforecasts_ti, forecasts_national, directory=directory, prefix=prefix, forecast_date=forecast_date)
        
    def plot_forecasts(self, fluforecasts_ti, forecasts_national, directory=".", prefix="", forecast_date=None, mode="flusight"):
        forecast_date_str=str(forecast_date)
        if forecast_date == None:
            forecast_date = self.mask_date
        if forecasts_national is None:
            if mode == "metrocast":
                forecasts_national = fluforecasts_ti.sum(axis=-1)
            else:
                raise ValueError("forecasts_national is required for mode='flusight'")
        idx_now = self.inpaintfrom_idx-1
        idx_horizon = idx_now+4

        plot_specs = {"all" : {
                                "quantiles_idx":range(11),
                                "color":"lightcoral",
                                },
                        "50-95" : {
                                "quantiles_idx":[1, 6],
                                "color":"darkblue"
                                }
                    }

        color_gt = "black"
        color_past='grey'
        y_label = "New Hosp. Admissions" if mode != "metrocast" else "ED visits pct"
        national_title = "National" if mode != "metrocast" else "Aggregate"
        median_q = 0.5 if 0.5 in myutils.flusight_quantiles else myutils.flusight_quantiles[12]

        nplace_toplot = len(self.season_setup.locations)
        #nplace_toplot = 3 # less plots for faster iteration
        plot_past_median = False
        if plot_past_median:
            plotrange=slice(None)
        else:
            plotrange=slice(self.inpaintfrom_idx,-1)


        #if self.season_first_year == "2023" or self.season_first_year == "2024":
        #    gt2022 = GroundTruth(season_first_year="2022", 
        #                    data_date=datetime.datetime.combine(datetime.date(2023,7,15), datetime.datetime.min.time()),
        #                    mask_date=datetime.datetime.today(),
        #                    channels=self.channels,
        #                    image_size=self.image_size,
        #                    payload=pd.read_csv("custom_datasets/nc_payload_gt.csv", parse_dates=["week_enddate"]))
        #if self.season_first_year == "2024":
        #    gt2023 = GroundTruth(season_first_year="2023", 
        #        data_date=datetime.datetime.combine(datetime.date(2023,7,15), datetime.datetime.min.time()),
        #        mask_date=datetime.datetime.today(),
        #        channels=self.channels,
        #        image_size=self.image_size,
        #        payload=pd.read_csv("custom_datasets/nc_payload_gt.csv", parse_dates=["week_enddate"]))

        for plot_title, plot_spec in plot_specs.items():
            #print(f"doing {plot_title}...")
            fig, axes = plt.subplots(nplace_toplot+1, 2, figsize=(10,nplace_toplot*3.5), dpi=200)
            for iax in range(2):
                ax = axes[0][iax]
    
                x = np.arange(self.image_size)
                if iax == 0:
                    x_lims_idx = (0, 51)
                    x_lims = (pd.to_datetime(self.gt_xarr["date"][x_lims_idx[0]].values), 
                            pd.to_datetime(self.gt_xarr["date"][x_lims_idx[1]].values))
                elif iax == 1:
                    x_lims_idx = (idx_now-3, idx_horizon)
                    x_lims = (pd.to_datetime(self.gt_xarr["date"][x_lims_idx[0]].values), 
                            pd.to_datetime(self.gt_xarr["date"][x_lims_idx[1]].values))
                # US WIDE: quantiles and median, US-wide
                for iqt in plot_spec["quantiles_idx"]:
                    #print(f"up: {flusight_quantile_pairs[iqt,0]} - lo: {flusight_quantile_pairs[iqt,1]}")
                    # TODO: not exactly true that it is the sum of quantiles (sum of quantile is not quantile of sum)
                    ylo = np.quantile(forecasts_national, myutils.flusight_quantile_pairs[iqt,0], axis=0)[0]
                    yup = np.quantile(forecasts_national, myutils.flusight_quantile_pairs[iqt,1], axis=0)[0]
                    ax.fill_between(self.gt_xarr["date"][plotrange], 
                                    ylo[plotrange], 
                                    yup[plotrange], 
                                    alpha=.1, 
                                    color=plot_spec["color"])
    
                    # widest quantile pair is the first one. We take the up quantile of it + a few % as x_lim
                    if iqt == plot_spec["quantiles_idx"][0]:
                        if plot_past_median:
                            max_y_value = max(yup[x_lims_idx[0]:x_lims_idx[1]])
                        else:
                            max_y_value = max(yup[self.inpaintfrom_idx:x_lims_idx[1]])
                        max_y_value = max(max_y_value, self.gt_xarr.data[0,:self.inpaintfrom_idx].sum(axis=1)[x_lims_idx[0]:x_lims_idx[1]].max())
                        max_y_value = max_y_value + max_y_value*.05 # 10% more
    
                # median
                ax.plot(
                    self.gt_xarr["date"][plotrange],
                    np.quantile(forecasts_national, median_q, axis=0)[0][plotrange],
                    color=plot_spec["color"],
                    marker=".",
                    label="forecast median",
                )
    
                # ground truth
                ax.plot(self.gt_xarr["date"][:self.inpaintfrom_idx],
                        self.gt_xarr.data[0,:self.inpaintfrom_idx].sum(axis=1), color=color_gt, marker = '.', lw=.5, label='ground-truth')
                ax.plot(self.gt_xarr["date"][self.inpaintfrom_idx:],
                        self.gt_xarr.data[0,self.inpaintfrom_idx:].sum(axis=1), 
                        color='red', 
                        marker = '.', 
                        lw=.1, 
                        label='ground-truth',
                        markersize=.4)

                #if self.season_first_year == "2023" or self.season_first_year == "2024":
                #    ax.plot(gt2022.gt_xarr.data[0,:].sum(axis=1), color=color_past, ls='dashed', lw=.5, label='2022 ground-truth')
                #if self.season_first_year == "2024":
                #    ax.plot(gt2022.gt_xarr.data[0,:].sum(axis=1), color=color_past, ls='dashdot', lw=.5, label='2023 ground-truth')

                if iax==0:
                    ax.legend(fontsize=8)
    
                #ax.set_xticks(np.arange(0,53,13))


                ax.set_xlim(x_lims)
                ax.set_ylim(bottom=0, top=max_y_value)
                ax.axvline(self.gt_xarr["date"][idx_now].values, c='k', lw=1, ls='-.')
                if iax == 0:
                    ax.axvline(self.gt_xarr["date"][idx_horizon].values, c='k', lw=1, ls='-.')
                ax.set_title(national_title)

                sns.despine(ax = ax, trim = True, offset=4)

                # INDIVDIDUAL STATES: quantiles, median and ground-truth
                max_y_value = np.zeros(nplace_toplot)
                for iqt in plot_spec["quantiles_idx"]:
                    yup = np.quantile(fluforecasts_ti, myutils.flusight_quantile_pairs[iqt,0], axis=0)[0]
                    ylo = np.quantile(fluforecasts_ti, myutils.flusight_quantile_pairs[iqt,1], axis=0)[0]

                    # widest quantile pair is the first one. We take the up quantile of it + a few % as x_lim
                    if iqt == plot_spec["quantiles_idx"][0]:
                        for ipl in range(nplace_toplot):
                            if plot_past_median:
                                max_y_value[ipl] = max(ylo[x_lims_idx[0]:x_lims_idx[1], ipl])
                            else:
                                max_y_value[ipl] = max(ylo[self.inpaintfrom_idx:x_lims_idx[1], ipl])
                            #max_y_value[ipl] =  max(ylo[x_lims[:x_lims[1], ipl])
                            max_y_value[ipl] = max(max_y_value[ipl], self.gt_xarr.data[0,:self.inpaintfrom_idx, ipl][x_lims_idx[0]:x_lims_idx[1]].max())
                            max_y_value[ipl] = max_y_value[ipl] + max_y_value[ipl]*.05 # 10% more for the y_max value

                    for ipl in range(nplace_toplot):
                        ax = axes[ipl+1][iax]
                        ax.fill_between(self.gt_xarr["date"][plotrange],  (yup[:,ipl])[plotrange], (ylo[:,ipl])[plotrange], alpha=.1, color=plot_spec["color"])

                # median line and ground truth for states
                for ipl in range(nplace_toplot):
                    location_name=self.season_setup.get_location_name(self.season_setup.locations[ipl])
                    ax = axes[ipl+1][iax]
                    # median
                    ax.plot(
                        self.gt_xarr["date"][plotrange],
                        np.quantile(fluforecasts_ti, median_q, axis=0)[0, :, ipl][plotrange],
                        color=plot_spec["color"],
                        marker=".",
                        lw=.5,
                    )
                    # ground truth
                    ax.plot(self.gt_xarr["date"][:self.inpaintfrom_idx],
                            self.gt_xarr.data[0,:self.inpaintfrom_idx, ipl], color=color_gt, marker = '.', lw=.5)
                    ax.plot(self.gt_xarr["date"][self.inpaintfrom_idx:],
                            self.gt_xarr.data[0,self.inpaintfrom_idx:, ipl], color='red', marker = '.', lw=.1, markersize=.4)

                    for hist_season, hist_dates, hist_values in self._get_historical_series(
                        self.season_setup.locations[ipl]
                    ):
                        ax.plot(
                            hist_dates,
                            hist_values,
                            color=color_past,
                            ls="dashed",
                            lw=.5,
                            label=f"{hist_season}",
                        )

                    ax.axvline(self.gt_xarr["date"][idx_now].values, c='k', lw=1, ls='-.')
                    if iax == 0:
                        ax.axvline(self.gt_xarr["date"][idx_horizon].values, c='k', lw=1, ls='-.')
                    ax.set_xlim(x_lims)
                    ax.set_ylim(bottom=0, top=max_y_value[ipl])
                    if iax == 0:
                        ax.set_ylabel(y_label)
                    ax.set_title(location_name)
                    # rotate the x axis labels
                    ax.tick_params(axis='x', rotation=45)
                    #print the tick label as 12 J-22
                    import matplotlib.dates as mdates
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b-%y'))

                    sns.despine(ax = ax, trim = True, offset=4)
            fig.tight_layout()
            plt.savefig(f"{directory}/{prefix}-{forecast_date_str}-plot{plot_title}.pdf")

    def export_forecasts_2023(self, fluforecasts_ti, forecasts_national=None, directory=".", prefix="", forecast_date=None, save_plot=True, nochecks=False, rate_trend=True, mode="flusight"):
        forecast_date_str=str(forecast_date)
        if forecast_date == None:
            forecast_date = self.mask_date
        season_start_date = datetime.date(int(self.season_first_year), self.season_setup.season_start_month, self.season_setup.season_start_day)

        reference_date = pd.to_datetime(forecast_date).date()
        reference_date_str = str(reference_date)
        base_index = pd.date_range(
            season_start_date,
            season_start_date + datetime.timedelta(days=self.image_size * 7),
            freq="W-SAT",
        )
        target_dates = [reference_date + datetime.timedelta(days=7 * h) for h in range(4)]
        target_dates = pd.to_datetime(target_dates)
        horizon_map = {pd.to_datetime(d): h for h, d in enumerate(target_dates)}

        df_list = []
        for qt in myutils.flusight_quantiles:
            a = pd.DataFrame(
                np.quantile(
                    fluforecasts_ti[:, :, :, : len(self.season_setup.locations)], qt, axis=0
                )[0],
                columns=self.season_setup.locations,
                index=base_index,
            ).loc[target_dates]

            a = a.reset_index().rename(columns={"index": "target_end_date"})
            a = pd.melt(a, id_vars="target_end_date", var_name="location")
            a["output_type_id"] = "{:.3f}".format(qt).rstrip("0").rstrip(".")
            df_list.append(a)

        df = pd.concat(df_list, ignore_index=True)

        if mode == "metrocast":
            df["reference_date"] = reference_date_str
            df["output_type"] = "quantile"
            df["horizon"] = df["target_end_date"].map(horizon_map)
            df["target"] = np.where(
                df["location"] == "nyc", "ILI ED visits pct", "Flu ED visits pct"
            )
            df = df[
                [
                    "reference_date",
                    "target",
                    "horizon",
                    "target_end_date",
                    "location",
                    "output_type",
                    "output_type_id",
                    "value",
                ]
            ]

            if not nochecks:
                assert sum(df["value"] < 0) == 0
                assert sum(df["value"].isna()) == 0

            df.to_csv(f"{directory}/{reference_date_str}-{prefix}.csv", index=False)
            if save_plot:
                if forecasts_national is None:
                    forecasts_national = fluforecasts_ti.sum(axis=-1)
                self.plot_forecasts(
                    fluforecasts_ti,
                    forecasts_national,
                    directory=directory,
                    prefix=prefix,
                    forecast_date=forecast_date,
                    mode=mode,
                )
            return

        if forecasts_national is None:
            raise ValueError("forecasts_national is required for mode='flusight'")

        target_dict = {d: f"{h}" for d, h in horizon_map.items()}
        updated_df_list = []
        for qt, dfd in zip(myutils.flusight_quantiles, df_list):
            us_vals = pd.DataFrame(
                np.quantile(forecasts_national, qt, axis=0)[0],
                columns=["US"],
                index=base_index,
            ).loc[target_dates]
            us_vals = us_vals.reset_index().rename(columns={"index": "target_end_date"})
            us_vals = pd.melt(us_vals, id_vars="target_end_date", var_name="location")
            dfd = pd.concat([dfd, us_vals], ignore_index=True)
            updated_df_list.append(dfd)

        df = pd.concat(updated_df_list, ignore_index=True)
        df["reference_date"] = forecast_date_str
        df["target"] = "wk inc flu hosp"
        df["horizon"] = df["target_end_date"].map(target_dict)
        df["output_type"] = "quantile"
        df = df[["reference_date","target","horizon","target_end_date","location","output_type","output_type_id","value"]]
        df

        # Suppress verbose output for column information  
        # for col in df.columns:
        #     print(col)
        #     print(df[col].unique())

        if not nochecks:
            assert sum(df["value"]<0) == 0
            assert sum(df["value"].isna()) == 0

        # check for Error when validating format: Entries in `value` must be non-decreasing as quantiles increase:
        for tg in target_dates:
            old_vals = np.zeros(len(self.season_setup.locations)+1)
            for dfd in df_list:  # very important to not call this df: it overwrites in namesapce the exported df
                new_vals = dfd[dfd["target_end_date"]==tg]["value"].to_numpy()
                if not (new_vals-old_vals >= 0).all():
                    num_negative = sum((new_vals-old_vals) < 0)
                    print(f" !!!! Quantile validation failed: {num_negative} negative values on {tg}")
                else:
                    pass
                    #print(f"""ok for {dfd["quantile"].unique()}, {tg}""")
                old_vals = new_vals

#        if rate_trend:
#            df_list=[]
#            for sim_id in np.arange(fluforecasts_ti.shape[0]):
#            #for qt in myutils.flusight_quantiles:
#                a =  pd.DataFrame(fluforecasts_ti[:,:,:,:len(self.season_setup.locations)], 
#                        columns= self.season_setup.locations, index=pd.date_range(self.season_setup.fluseason_startdate, self.season_setup.fluseason_startdate + datetime.timedelta(days=64*7), freq="W-SAT")).loc[target_dates]
#                a["US"] = pd.DataFrame(forecasts_national[sim_id],
#                        columns= ["US"], index=pd.date_range(self.season_setup.fluseason_startdate, self.season_setup.fluseason_startdate + datetime.timedelta(days=64*7), freq="W-SAT")).loc[target_dates]
#
#                a = a.reset_index().rename(columns={'index': 'target_end_date'})
#                a = pd.melt(a,id_vars="target_end_date",var_name="location")
#
#                
#                df_list.append(a)
#
#            df2 = pd.concat(df_list)
#            df2["reference_date"] = forecast_date_str
#            df2["target"] = "wk flu hosp rate change"
#            df2["horizon"] = df["target_end_date"].map(target_dict)
#            df2["output_type"] = "pmf"
#            df2 = df2[["reference_date","target","horizon","target_end_date","location","output_type","output_type_id","value"]]


        df.to_csv(f"{directory}/{forecast_date_str}-{prefix}.csv", index=False)

        if save_plot:
            self.plot_forecasts(fluforecasts_ti, forecasts_national, directory=directory, prefix=prefix, forecast_date=forecast_date)
