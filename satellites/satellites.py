#!/usr/bin/env python

import argparse
import time

from urllib.request import urlopen
import json
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import math

import folium
from folium import plugins
import branca.colormap as cm

import staticmaps
import imageio
from PIL import ImageDraw 

import seaborn as sns
import warnings
import datetime as dt
from datetime import datetime, timezone


from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import leaders
from scipy.cluster.hierarchy import fcluster

#count twilight time
from skyfield import api, almanac

'''
GET LIST OF SATELLITES AND THEIR LOCATION
'''
def satellite_positions(test=False):
    url_satellites = "https://satellitemap.space/json/sl.json"
    response = urlopen(url_satellites)
    satellites_json = json.loads(response.read())
    if test: print(satellites_json['sats'][0])

    df = pd.DataFrame(satellites_json['sats'])
    df = df.convert_dtypes()
    df = df[df['lat'] != 0.0]
    df = df[df['lng'] != 0.0]
    
    if test: print(df)
    return df

'''
Distance calculation on top of earth
'''
def distance(origin, destination): #found here https://gist.github.com/rochacbruno/2883505
    lat1, lon1 = origin[0],origin[1]
    lat2, lon2 = destination[0],destination[1]
    radius = 6371 # km
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d

'''
CALCULATE DISTANCE BETWEEN ALL SATELLITES
TODO: Will be nice to parallel it, because now it takes around 180 seconds on 4340 satellites.
'''
def satellite_distances(df, test=False):
    list_sat=df[['lat','lng']].values.tolist()
    dist_satellite = np.empty([len(list_sat),len(list_sat)])
    start = timer()
    for i in range(0,len(list_sat)):
        dist_satellite[i][i]=np.nan
        for k in range(i+1, len(list_sat)):
            if test: print(i,k, end='\r')
            dist_satellite[i][k] = dist_satellite[k][i] = distance(list_sat[i],list_sat[k])

    if test: print('\n')
    end = timer()
    if test: print(end - start)

    return dist_satellite

#TODO: need to finish testing - https://saturncloud.io/blog/calculating-distance-matrix-between-rows-of-a-pandas-dataframe-with-latitude-and-longitude/
def satellite_distances_improved(df, test=False):
    import numpy as np

    # Convert latitude and longitude to radians
    df['Lat'] = np.deg2rad(df['Lat'])
    df['Lon'] = np.deg2rad(df['Lon'])

    from scipy.spatial.distance import cdist

    # Calculate the distance matrix
    dist_matrix = cdist(df[['Lat', 'Lon']], df[['Lat', 'Lon']], metric='haversine')

    # Convert to kilometers
    dist_matrix *= 6371.0

    # Convert the distance matrix to a DataFrame
    dist_df = pd.DataFrame(dist_matrix, index=df.index, columns=df.index)

    return None

'''
REDUCING SINGLE SATELLITES - CONGELATION CLEANUP
'''
def satellite_cleanup(df, dist_satellite, lim=100, neighbors=2, test=False):
    df['nearest'] = np.nanmin(dist_satellite,axis=1).tolist()
    df['neighbors'] = np.count_nonzero(np.asarray(dist_satellite)<lim,axis=1,keepdims=True)

    if test: print(df)

    plt.hist(df['nearest'])
    plt.title(f'Histogram - nearest {lim}')
    plt.savefig(f'./tmp/my_plot_nearest_hist_{lim}.png')
    plt.close()
    print(f'./tmp/my_plot_nearest_hist_{lim}.png created')

    plt.hist(df['neighbors'])
    plt.title(f'Histogram - neighbors {neighbors}x{lim}')
    plt.savefig(f'./tmp/my_plot_neighbors_hist_{neighbors}_{lim}.png')
    plt.close()
    print(f'./tmp/my_plot_neighbors_hist_{neighbors}_{lim}.png created')

    #df = df[df['nearest'] < lim]
    df = df[df['neighbors'] >= neighbors]
    if test: print(df)

    return df

'''
PRINT THEM ON THE MAP
'''
def satellites_map_html(df, home=[49.22,-122.65], name='./tmp/satellites.html', test=False):
    map=folium.Map(tiles="Stamen Toner")
    map.add_child(folium.Marker(location=home,popup='Home', icon=folium.Icon(color='green')))

    colormap = cm.StepColormap(colors=['green','yellow','red'],
                            index=[df['neighbors'].min(),df['neighbors'].median(),df['neighbors'].max()],
                            vmin=df['neighbors'].min(),
                            vmax=df['neighbors'].max())

    list_coor=df[['id','lat','lng']].values.tolist()
    for i in list_coor:
        map.add_child(folium.CircleMarker(location=[i[1],i[2]],
                                    popup=i[0],radius=1))
        
    for loc, p in zip(zip(df['lat'],df['lng']),df['neighbors']):
        folium.Circle(
        location=loc,
        radius=1,
        fill=True, 
        color=colormap(p)
        ).add_to(map)

    map.save(name)
    if test: print(f'{name} created')

def satellites_map_png(df, home=[49.22,-122.65], name='./tmp/satellites.png', size=2048, test=False):
    context = staticmaps.Context()
    context.set_tile_provider(staticmaps.tile_provider_StamenToner)
    context.set_center(staticmaps.parse_latlng("0,0"))
    context.set_zoom(3)

    list_coor=df[['lat','lng']].values.tolist()
    for i in list_coor:
        point = staticmaps.create_latlng(i[0], i[1])
        context.add_object(staticmaps.Marker(point, color=staticmaps.GREEN, size=10))
        #context.add_object(staticmaps.Circle(point, 10, fill_color=staticmaps.GREEN, color=staticmaps.YELLOW, width=1))

    image = context.render_pillow(size, size)
    ImageDraw.Draw(image).text((10, 10),f'{name}',(255,255,255))
    image.save(f'{name}')
    if test: print(f'{name} created')

'''
ANALYTICS
'''
# found here: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def satellite_analytics_visualization(df, test=False):
    plt.boxplot(df['neighbors'])
    plt.figure(figsize=(25, 10))
    plt.savefig('./tmp/my_plot_neighbors_box.png')
    plt.close()
    print('./tmp/my_plot_neighbors_box.png created')

    plt.scatter(df["neighbors"], df["nearest"])
    plt.figure(figsize=(25, 10))
    plt.title("Scatter plot analysing neighbors vs nearest\n")
    plt.xlabel("neighbors", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Black'})
    plt.ylabel("nearest", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Black'} )
    plt.savefig('./tmp/my_plot_neighbors_vs_nearest_box.png')
    plt.close()
    print('./tmp/my_plot_neighbors_vs_nearest_box.png created')

'''
Finding first real head for each leader and returning with L, M, H
'''
### loop through L
def node_unleash(L, Z, cluster_hight, heads=np.array([], dtype=int), deep=False):
    for node in L:
        if node >= cluster_hight:
            L1=[int(Z[node-cluster_hight][0]),int(Z[node-cluster_hight][1])]
            heads = node_unleash(L1, Z, cluster_hight, heads, deep=True)
            
            if deep: break
        else:
            heads = np.append(heads, node)
            break

    return heads

### leaders replacement
def fcluster_leaders_with_heads(Z, clusters, test=False):
    L, M = leaders(Z, clusters)
    H = node_unleash(L,Z,clusters.shape[0])

    #if test: 
    #    print(f'Z: {Z} {type(Z)}')
    #    print(f'clusters: {clusters} {type(clusters)}')
    #    print(f'L: {L} {type(L)}')
    #    print(f'M: {M} {type(M)}')
    #    print(f'H: {H} {type(H)}')

    return L, M, H

'''
Cluster visualization 
'''
def satellite_preclustering_analyze_visualization(df, test=False):
    #printing locations
    name='scatter'
    plt.figure(figsize=(25, 10))
    plt.scatter(df['lng'], df['lat'])
    plt.savefig(f'./tmp/{name}.png')
    plt.close()
    print(f'./tmp/{name}.png created')

    #printing full chart
    name='dendrogram'
    Z = linkage(df[['lng','lat']], 'ward')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig(f'./tmp/{name}.png')
    plt.close()
    print(f'./tmp/{name}.png created')

    #TODO: automate search of this value
    max_d = 50
    name='dendrogram_trancated'
    Z = linkage(df[['lng','lat']], 'ward')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram Trancated')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    fancy_dendrogram(
        Z,
        #truncate_mode='lastp',  # show only the last p merged clusters
        #p=20,  # show only the last p merged clusters
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        #show_contracted=True,  # to get a distribution impression in truncated branches
        annotate_above=10,  # useful in small plots so annotations don't overlap
        max_d=max_d,  # plot a horizontal cut-off line
    )
    plt.savefig(f'./tmp/{name}.png')
    plt.close()
    print(f'./tmp/{name}.png created')

    name='clusters'
    clusters = fcluster(Z, max_d, criterion='distance')
    L, M, H = fcluster_leaders_with_heads(Z, clusters, test=test)

    #TODO: for each satellite in cluster, add number of cluster, plus count of satellites in this cluster
    df['cluster']=clusters.transpose()
    satellites_in_cluster = df.groupby('cluster')['cluster'].count()

    #if test: print(f"satellites_in_cluster: {satellites_in_cluster}")
    
    df['satellites_in_cluster'] = df['cluster'].transform(lambda x: satellites_in_cluster[x])    

    plt.figure(figsize=(25, 10))
    plt.scatter(df['lng'], df['lat'], c=clusters, cmap='prism')  # plot points with cluster dependent colors
    plt.scatter(df.loc[df.index[H],'lng'], df.loc[df.index[H],'lat'], c='black', s=200, alpha=0.5);
    plt.savefig(f'./tmp/{name}.png')
    plt.close()
    print(f'./tmp/{name}.png created')

    return df.loc[df.index[H]]

# twilight calculation
def twilight_status(time_utc=1690795568537, home=[49.22,-122.65], test=False):
    from skyfield import api, almanac
    from datetime import datetime, timedelta, timezone
    import pytz
    from skyfield.api import wgs84
    
    tz_user=timezone_by_coordinates(home=home, test=test)
    
    topos = wgs84.latlon(home[0], home[1])
    
    planets = api.load('de421.bsp')
    f = almanac.dark_twilight_day(planets, topos)
    
    ts = api.load.timescale()
    t0 = ts.from_datetime(time_utc)
    status_result = f(t0)

    if status_result in [1,2,3]:
        if t0.astimezone(tz_user).hour > 12:
            status_result = status_result + 4

    return status_result

def twilight_status_name(status_result, test=False):
    status=['0 — Dark of night.',                           #0
            '1 — Astronomical twilight in the morning.',    #1
            '2 — Nautical twilight in the morning.',        #2
            '3 — Civil twilight in the morning.',           #3
            '4 — Sun is up.',                               #4
            '11 — Astronomical twilight at night.',         #5
            '12 — Nautical twilight at night.',             #6
            '13 — Civil twilight at night.',]               #7
    
    return f"{status[status_result]}"

def timezone_by_coordinates(home=[49.22,-122.65], test=False):
    import timezonefinder, pytz

    tf = timezonefinder.TimezoneFinder()

    timezone_str = tf.certain_timezone_at(lat=home[0], lng=home[1])

    if timezone_str is None:
        raise ValueError("Could not determine the time zone")
    else:
        return pytz.timezone(timezone_str)
    
#this information is correct, provided in user timezone, but have only sunrise and sunset, instead of twilight information
def satellite_sunlight(id=57066, home=[49.22,-122.65], test=False):
    url_satellite_passes = f'https://satellitemap.space/api/passes?norad={id}&lat={home[0]}&lng={home[1]}'
    response = urlopen(url_satellite_passes)
    satellite_passes_json = json.loads(response.read())
    if test: print(satellite_passes_json['data']['sunlight'])

def satellite_passes(id=57066, home=[49.22,-122.65], test=False):
    from datetime import datetime, timedelta, timezone
    import pytz

    #https://satellitemap.space/api/passes?norad=57061&lat=49.22&lng=-122.65
    url_satellite_passes = f'https://satellitemap.space/api/passes?norad={id}&lat={home[0]}&lng={home[1]}'
    response = urlopen(url_satellite_passes)
    satellite_passes_json = json.loads(response.read())

    if len(satellite_passes_json['data']['passes']) == 0:
        raise ValueError(f"{id} has no overpasses for next 3 days")

    if test: print(satellite_passes_json['data']['passes'][0])

    df = pd.DataFrame(satellite_passes_json['data']['passes'])
    df = df.convert_dtypes()

    try:
        tz_user=timezone_by_coordinates(home=home, test=test)
    except ValueError as e:
        print(e)
        return []
    
    df['time_local'] = pd.to_datetime(df['utcpass']/1000, unit='s').dt.tz_localize(tz_user)
    df['time_utc'] = pd.to_datetime(df['utcpass']/1000, unit='s').dt.tz_localize(tz_user).dt.tz_convert(tz=pytz.utc)
    df['status_correction'] = df.apply(lambda x: twilight_status(time_utc=x['time_utc'], home=home), axis=1)
    df['status_description'] = df.apply(lambda x: twilight_status_name(status_result=x['status_correction']), axis=1)

    if test: print(df)

    return df

def main(args):
    test=args.test
    
    ### GETTING POSITION OF SATELLITES, FROM FILE OR FROM SATELLITE MAP
    if args.load_original:
        df = pd.read_pickle(args.load_original)
    else:
        df = satellite_positions(test=test)
    
    if args.save_original: 
        df.to_pickle(args.save_original)

    ### GETTING DISTANCES BETWEEN ALL SATELLITES, FROM FILE OR CALCULATING
    if args.load_distances:
        with open(args.load_distances, 'rb') as f:
            dist_satellite = np.load(f)
    else:
        dist_satellite = satellite_distances(df, test=test)

    if args.save_distances:
        with open(args.save_distances, 'wb') as f:
            np.save(f, dist_satellite)

    if args.home_lat and args.home_lng: 
        home=[args.home_lat, args.home_lng]
    else:
        home=[49.22,-122.65]

    ### CLEANING DATA TO GET ONLY CONCILIATIONS, repeating 10 times (probably some better way exist)
    df_clean = df
    frames = []

    for i in range (1, 11):
        ### a little bit of visualization
        satellites_map_png(df_clean, name=f'./tmp/satellites-{i}.png', test=test)
        image = imageio.v2.imread(f'./tmp/satellites-{i}.png')
        frames.append(image)

        df_clean = satellite_cleanup(df_clean, dist_satellite, lim=i*100, neighbors=i+1, test=test)
        dist_satellite = satellite_distances(df_clean, test=test)

        #time.sleep(10)
    
    satellites_map_html(df_clean, test=test)
    imageio.mimsave('./tmp/satellites.gif', frames, duration = 200, loop=0)
    #satellite_analytics_visualization(df_clean, test=test)
    df_conciliation_heads=satellite_preclustering_analyze_visualization(df_clean, test=test)

    ### FOR EACH CONCILIATION COUNTING NEXT OVERPASS
    #if test: print(df_conciliation_heads)
    #if test: print(df_conciliation_heads.info())

    final_result = pd.DataFrame()
    
    for index, row in df_conciliation_heads.iterrows():
        if test: print(f"Row: {row} {type(row)}")
        
        try:
            df_times = satellite_passes(id=row['id'], home=home, test=test)
            for i, v in row.items():
                df_times[i] = v
            # if test: 
            #     print(df_times)
            #     print(df_times.info())

            final_result = final_result._append(df_times[['id',
                                               'cluster',
                                               'satellites_in_cluster',
                                               'elazf',
                                               'elazt',
                                               'duration_mins',
                                               'time_local',
                                               'status_correction',
                                               'status_description']],
                                                ignore_index = True)
        except ValueError as e:
            print(e)
    
    if test:
        print(final_result)

    #cleanup
    final_result['from'] = final_result.apply(lambda x: x['elazf'][1], axis=1)
    final_result['to'] = final_result.apply(lambda x: x['elazt'][1], axis=1)
    final_result = final_result.drop(columns=['elazf', 'elazt'])

    #filter
    final_result = final_result[final_result['satellites_in_cluster'] >= 10]
    final_result = final_result[final_result['duration_mins'] >= 3]
    final_result = final_result[final_result['status_correction'].isin([1,2,3,5,6,7])]

    print(final_result.sort_values(by=['time_local']))

    ### Output:
    # {conciliation_count} satellites will fly over you
    #   from {ekazf[1]} to {elazf[1]} for period of {duration_mins} minutes on {datetime} during {status_description}
    #   ...
    #   from {ekazf[1]} to {elazf[1]} for period of {duration_mins} minutes on {datetime} during {status_description}

    
if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description='Getting current location of StarLink satellites, \n'
                                            'getting out consolidation and trying to find time \n'
                                            'when it passes on top of you.', 
                                            formatter_class=argparse.RawTextHelpFormatter)
    
    argParser.add_argument("--test", required=False, 
                           help="run in test mode", action=argparse.BooleanOptionalAction)
    
    argParser.add_argument('--save-original', required=False, 
                           help="Save a pandas dataframe with original satellite information to file")
    argParser.add_argument('--load-original', required=False, 
                           help="Load a pandas dataframe with original satellite information from file")
    
    argParser.add_argument('--save-distances', required=False, 
                           help="Save a numpy dataframe with original satellite information to file")
    argParser.add_argument('--load-distances', required=False, 
                           help="Load a numpy dataframe with original satellite information from file")
    
    argParser.add_argument('--home-lat', required=False, 
                           help="Latitude of location to count overpass")
    
    argParser.add_argument('--home-lng', required=False, 
                           help="Longitude of location to count overpass")

    args = argParser.parse_args()

    if args.save_distances and args.save_original is None:
        argParser.error("--save-distances requires --save-original")
    if args.load_distances and args.load_original is None:
        argParser.error("--load-distances requires --load-original")

    if args.home_lat and args.home_lng is None:
        argParser.error("--home-lat requires --home-lng")
    if args.home_lat is None and args.home_lng:
        argParser.error("--home-lng requires --home-lat")

    main(args) 
