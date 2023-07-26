### Description:

Create a list of passes of Starlink satellites on top of your location

You already can find this information on the following site: https://satellitemap.space/

But you need to click on all visual constellation of Starlink satellites and get calculation for them.

You can get a list of all satellites at this moment like this:
```
$ curl 'https://satellitemap.space/json/sl.json' -H 'accept: application/json,*/*'  --compressed
```

You can get a list of passes for each satellite like this:
```
curl 'https://satellitemap.space/api/passes?norad=56803&lat=49.22&lng=-122.65' \
  -H 'accept: application/json,*/*' \
  --compressed
```

### OPTIONS
```
$ ./satellites.py -h
usage: satellites.py [-h] [--test | --no-test] [--save-original SAVE_ORIGINAL] [--load-original LOAD_ORIGINAL]
                     [--save-distances SAVE_DISTANCES] [--load-distances LOAD_DISTANCES]

Getting current location of StarLink satellites, 
getting out consolidation and trying to find time 
when it passes on top of you.

optional arguments:
  -h, --help            show this help message and exit
  --test, --no-test     run in test mode
  --save-original SAVE_ORIGINAL
                        Save a pandas dataframe with original satellite information to file
  --load-original LOAD_ORIGINAL
                        Load a pandas dataframe with original satellite information from file
  --save-distances SAVE_DISTANCES
                        Save a numpy dataframe with original satellite information to file
  --load-distances LOAD_DISTANCES
                        Load a numpy dataframe with original satellite information from file
```

### TODO:

- [x] need to visualize them on the map
- [ ] need to group them and get congelation
- [ ] for any satellite in constellation get passes
