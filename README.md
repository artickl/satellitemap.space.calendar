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

### TODO:

- [x] need to visualize them on the map
- [ ] need to group them and get congelation
- [ ] for any satellite in constellation get passes
