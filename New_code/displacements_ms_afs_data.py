import argparse
import os

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

import utilities


def initialize_hdf5(filepath):
    """ Initialize an empty HDF5 file with the basic structure. """
    with h5py.File(filepath, 'w') as f:
        f.attrs['description'] = 'Stations displacement data for multiple main shocks and their aftershocks data'


def filter_gps_stations(earthquakes, ngl_list, search_radius, earth_radius, min_station_number=3):
    stations_to_download = set()
    mainshocks_stations = {}
    mainshocks_days = {}

    for id, seq in earthquakes.groupby('seq_id'):
        if len(seq) <= 1:
            continue
        mainshock = seq[seq['type'] == 1]
        if mainshock.empty:
            continue

        max_radius = max(10 ** (mainshock.mag.values[0] / 2 - 0.79), search_radius)
        distances = haversine_distances(np.radians(ngl_list[["lat", "lon"]]),
                                        np.radians(mainshock[["lat", "lon"]].values))[:, 0]
        valid_mask = (earth_radius * distances <= max_radius) & (ngl_list.begin.values <= mainshock.day.values[0]) & (
                ngl_list.end.values >= mainshock.day.values[0])

        valid_stations = ngl_list.name.values[valid_mask]
        if len(valid_stations) >= min_station_number:
            stations_to_download.update(valid_stations)
            mainshocks_stations[id] = list(valid_stations)
            mainshocks_days[id] = mainshock.day.values[0].astype('datetime64[D]')

    return stations_to_download, mainshocks_stations, mainshocks_days


def download_and_filter_gps_data(stations_to_download, gps_data_output_path, mainshock_stations, mainshock_day,
                                 n_days_before_mainshock, n_days_after_mainshock):
    mainshock_data = {}

    for station in stations_to_download:
        file_path = os.path.join(gps_data_output_path, f"{station}.csv")
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, sep=" ", parse_dates=['date'])
            print(f"{station} data loaded.")
        else:
            try:
                data = utilities.get_ngl_gps_data(station, "IGS14", "tenv3")
                data.to_csv(file_path, sep=" ", index=False)
                print(f"{station} data downloaded and saved.")
                break
            except Exception as e:
                print(f"Error downloading data for station {station}: {e}")
                continue

        data['lon'] = data['lon'] % 360

        # Filter data for the days around each main shock this station is associated with
        for id in mainshock_stations.keys():
            if station in mainshock_stations[id]:
                start_date = mainshock_day[id] - np.timedelta64(n_days_before_mainshock, 'D')
                end_date = mainshock_day[id] + np.timedelta64(n_days_after_mainshock, 'D')
                subdata = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

                if len(subdata) == n_days_before_mainshock + n_days_after_mainshock + 1 and np.isfinite(
                        subdata[['lat', 'lon', 'height']].values).all():
                    if id not in mainshock_data:
                        mainshock_data[id] = []
                    mainshock_data[id].append(subdata)
                else:
                    print(f"Data for station {station} around main shock {id} is incomplete or contains NaNs.")

    return mainshock_data


def process_and_save_main_shocks_data(mainshock_data, earthquakes, ngl_list, earth_radius, after_shock_time_window, hdf5_path):
    conv_factor = earth_radius * 1e3 * np.pi / 180  # Convert from lat, lon (degrees) to meters

    for id, station_data in mainshock_data.items():
        velocities = []
        stations_positions = []

        for md in station_data:
            station_position = ngl_list[ngl_list.name == md['site'].iloc[0]][['lat', 'lon']].values[0]
            pos = md[['lat', 'lon', 'height']].values
            vel = np.diff(pos, axis=0)  # Calculate 1-day velocity: diff of position between 2 days
            vel[:, [0, 1]] *= conv_factor
            velocities.append(vel[:, np.newaxis, :])
            stations_positions.append(station_position[np.newaxis, :])

        if velocities:
            velocities = np.concatenate(velocities, axis=1)
            stations_positions = np.concatenate(stations_positions)

            # Extract aftershocks information
            seq = earthquakes[earthquakes.seq_id == id]
            mainshock = seq[seq['type'] == 1]
            aftershocks = seq[(seq['type'] == 2) & (seq['day'] > mainshock.day.values) & (
                        seq['day'] <= mainshock.day.values + after_shock_time_window)]
            #mainshock_location = mainshock[['lat', 'lon']].values[0]
            mainshock_location = mainshock[['lat', 'lon']].value
            aftershocks_locations = aftershocks[['lat', 'lon']].values
            aftershocks_mags = aftershocks['mag'].values

            if len(aftershocks_locations) > 1:
                # Save to HDF5
                with h5py.File(hdf5_path, 'a') as f:
                    grp = f.create_group(str(id))
                    grp.attrs['main_shock_day'] = mainshock.day.values[0].astype('datetime64[D]')
                    grp.attrs['main_shock_magnitude'] = mainshock.mag.values[0]
                    grp.attrs['main_shock_location'] = mainshock_location

                    grp.create_dataset('gps_displacements', data=velocities)
                    grp.create_dataset('stations_positions', data=stations_positions)
                    grp.create_dataset('aftershocks_magnitudes', data=aftershocks_mags)
                    grp.create_dataset('aftershocks_locations', data=aftershocks_locations)


def main():
    parser = argparse.ArgumentParser(description="Parameters to define to build your custom displacement data")
    parser.add_argument("earthquakes_path", type=str, help="Path to the earthquake catalog file")
    parser.add_argument("hdf5_file_output_path", type=str, help="Path to the HDF5 output file")
    parser.add_argument("gps_data_output_path", type=str, help="Path to store the GPS data")
    parser.add_argument("--search_radius", type=int, default=300, help="Search radius of GPS stations in km")
    parser.add_argument("--min_stations_per_main_shock", type=int, default=3,
                        help="Minimum number of GPS stations around a mainshock to consider it")
    # parser.add_argument("--min_main_shock_magnitude", type=float, default=6, help="Minimum mainshock magnitude")
    # parser.add_argument("--min_after_shock_magnitude", type=float, default=4, help="Minimum aftershock magnitude")
    parser.add_argument("--after_shock_time_window", type=int, default=45,
                        help="Days after the mainshock to search for aftershocks")
    parser.add_argument("--n_days_before_mainshock", type=int, default=7,
                        help="Number of days before the mainshock to consider")
    parser.add_argument("--n_days_after_mainshock", type=int, default=7,
                        help="Number of days after the mainshock to consider")
    args = parser.parse_args()

    initialize_hdf5(args.hdf5_file_output_path)
    earthquakes = pd.read_csv(args.earthquakes_path, sep=" ", parse_dates=['datetime'])
    earthquakes.rename(columns={'datetime': 'day'}, inplace=True)
    earthquakes['day'] = earthquakes.day.values.astype('datetime64[D]')
    earthquakes.sort_values(by='day', inplace=True)
    earthquakes.reset_index(drop=True, inplace=True)

    ngl_list = utilities.get_ngl_stations()
    earth_radius = 6371  # km
    stations_to_download, mainshocks_stations, mainshocks_days = filter_gps_stations(earthquakes, ngl_list,
                                                                                  args.search_radius, earth_radius, args.min_stations_per_main_shock)
    mainshocks_data = download_and_filter_gps_data(stations_to_download, args.gps_data_output_path, mainshocks_stations,
                                                  mainshocks_days, args.n_days_before_mainshock,
                                                  args.n_days_after_mainshock)

    process_and_save_main_shocks_data(mainshocks_data, earthquakes, ngl_list, earth_radius, args.ndays_after_mainshock,
                                     args.hdf5_file_output_path)


if __name__ == "__main__":
    main()
