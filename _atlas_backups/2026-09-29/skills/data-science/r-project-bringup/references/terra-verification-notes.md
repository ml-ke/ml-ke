# terra / raster verification notes

Pitfalls that make a correct-looking geospatial pipeline fail, or make a broken
one look like it worked.

## `compareGeom()` is strict about resolution, not just extent

`terra::compareGeom()` (used as a guard inside many package functions) compares
**crs, extent, resolution and ncell**. Two rasters that overlay "well enough"
visually can still fail it.

Worked example — a polygon rasterised on its own comes out off-grid:

```
land-cover raster : ext 52661.803, 741061.803, 8783175.672, 9140175.672
                   res 100 x 100            ncell 24,575,880
rasterise(sf poly): ext 52661.803, 741030.250, 8783175.672, 9140126.304
                   res 99.99542 x 99.98617   ncell 24,575,880

compareGeom(lc, rasterised_poly, stopOnError = TRUE)
#> Error: [compareGeom] extents do not match
```

The vector's bounding box drives the derived grid, so the resolution drifts (the
bbox spans a non-integer number of cells) and the right/top edges fall short of
the reference raster. Same crs, same dimensions, same ncell — still rejected.

**Consequences when verifying a package:**

- A function whose documented input is a shipped reference raster, but whose code
  also accepts a vector it rasterises internally, has two paths with different
  reliability. Use the shipped raster the examples and tests exercise.
- A bundled example raster and a land-cover raster from the same source often
  compare as `FALSE` under `all.equal(ext(...))` while still passing
  `compareGeom` — floating-point ties within tolerance. `compareGeom` is the
guard that matters; do not conclude a mismatch from a raw extent comparison.
- Diagnose with the numbers, not the error text:
  `crs(x, proj=TRUE)`, `paste(as.vector(ext(x)), collapse=", ")`, `res(x)`, `dim(x)`,
  `ncell(x)` for each object, side by side.

## Legends and time do not survive unless written properly

Functions that plot a categorical raster need the category table attached
(`terra::cats(x)[[1]]` non-NULL) and time-series helpers need the layer's time
attribute. Both live in the object, not the file, so:

- Attach them in memory (`add_legend_to_categorical_raster(...)`, `assign_time_period(...)`)
  before passing the raster on.
- When a report/document re-reads rasters from disk (`rast(params$path)`), write the
  **annotated** objects out with `writeRaster()` so cats and time are persisted —
  writing the raw source file path instead silently produces unlabelled output.
- A `.tif.aux.xml` sidecar next to the `.tif` is normal and carries the category
  metadata; do not delete it.

## Verify the numbers, not the plot

A rendered map proves the pipeline ran, not that it is right. Confirm with values
that can be checked by hand:

- row counts of the crosstab / long tables;
- the top-N change table as literal rows (source class → target class → area);
- `sum(df[["Freq"]])` against the cell count of the input raster;
- the count of distinct planning units against the attribute table of the zone raster.

Report those actual figures; "the pipeline ran and produced a Sankey diagram" is
not verification.
