# `rust-wkb`

This crate provides functions to convert `rust-geo` geometry types to and from [Well Known Binary](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry#Well-known_binary) format, i.e. [ISO 19125](https://www.iso.org/standard/40114.html)

## Examples

```rust
use geo_types::*;
use wkb::*;

let p: Geometry<f64> = Geometry::Point(Point::new(2., 4.));
let res = geom_to_wkb(&p);
assert_eq!(res, vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 16, 64]);
```

You can also 'read' a Geometry from a `std::io::Read`:

```rust
use geo_types::*;
use wkb::*;

let bytes: Vec<u8> = vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 16, 64];
let p: Geometry<f64> = wkb_to_geom(&mut bytes.as_slice()).unwrap();
assert_eq!(p, Geometry::Point(Point::new(2., 4.)));
```

Adding proper `*Ext` traits is planned.



