#!/usr/bin/env python

import argparse

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

import seaborn as sns
import warnings
import datetime as dt

#import sklearn
#from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


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

'''
REDUCING SINGLE SATELLITES - CONGELATION CLEANUP
'''
def satellite_cleanup(df, dist_satellite, lim=100, neighbors=2, test=False):
    df['nearest'] = np.nanmin(dist_satellite,axis=1).tolist()
    df['neighbors'] = np.count_nonzero(np.asarray(dist_satellite)<lim,axis=1,keepdims=True)

    if test: print(df)

    plt.hist(df['nearest'])
    plt.title('Histogram - nearest')
    plt.savefig('../tmp/my_plot_nearest_hist.png')
    plt.close()
    print('../tmp/my_plot_nearest_hist.png created')

    plt.hist(df['neighbors'])
    plt.title('Histogram - neighbors')
    plt.savefig('../tmp/my_plot_neighbors_hist.png')
    plt.close()
    print('../tmp/my_plot_neighbors_hist.png created')

    #df = df[df['nearest'] < lim]
    df = df[df['neighbors'] >= neighbors]
    if test: print(df)

    return df

'''
PRINT THEM ON THE MAP
'''
def satellites_map(df, home=[49.22,-122.65], test=False):
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

    map.save('../tmp/satellites.html')
    print('../tmp/satellites.html created')

'''
ANALYTICS
'''
def satellite_analytics(df, test=False):
    plt.boxplot(df['neighbors'])
    plt.savefig('../tmp/my_plot_neighbors_box.png')
    plt.close()
    print('../tmp/my_plot_neighbors_box.png created')


    plt.scatter(df["neighbors"], df["nearest"])
    plt.title("Scatter plot analysing neighbors vs nearest\n")
    plt.xlabel("neighbors", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Black'})
    plt.ylabel("nearest", fontdict={'fontsize': 12, 'fontweight' : 5, 'color' : 'Black'} )
    plt.savefig('../tmp/my_plot_neighbors_vs_nearest_box.png')
    plt.close()
    print('../tmp/my_plot_neighbors_vs_nearest_box.png created')

    new_df = df[['neighbors','nearest']]
    new_df.shape

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

    ### CLEANING DATA TO GET ONLY CONCILIATIONS
    df_clean = satellite_cleanup(df, dist_satellite, test=test)

    ### MAKING MAP OF CONCILIATIONS
    satellites_map(df_clean, test=test)

    ### ADDITIONAL TESTING ANALYTICS
    satellite_analytics(df_clean, test=test)

    ### TODO: FOR EACH CONCILIATION COUNTING NEXT OVERPASS
    # TODO: conciliation_overpass(df_clean, home=args.home, test=test)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description='Getting current location of StarLink satellites, \n'
                                            'getting out consolidation and trying to find time \n'
                                            'when it passes on top of you.', 
                                            formatter_class=argparse.RawTextHelpFormatter)
    
    argParser.add_argument("--test", help="run in test mode", action=argparse.BooleanOptionalAction)
    
    argParser.add_argument('--save-original', required=False, 
                           help="Save a pandas dataframe with original satellite information to file")
    argParser.add_argument('--load-original', required=False, 
                           help="Load a pandas dataframe with original satellite information from file")
    
    argParser.add_argument('--save-distances', required=False, 
                           help="Save a numpy dataframe with original satellite information to file")
    argParser.add_argument('--load-distances', required=False, 
                           help="Load a numpy dataframe with original satellite information from file")

    args = argParser.parse_args()

    if args.save_distances and args.save_original is None:
        argParser.error("--save-distances requires --save-original")
    if args.load_distances and args.load_original is None:
        argParser.error("--load-distances requires --load-original")

    main(args) 