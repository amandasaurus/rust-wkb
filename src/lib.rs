//! This crate provides functions to convert `rust-geo` geometry types to and from Well Known
//! Binary format.
//!
//! # Examples
//!
//! ```rust
//! # extern crate geo;
//! # extern crate wkb;
//! # fn main() {
//! use geo::*;
//! use wkb::*;
//!
//! let p: Geometry<f64> = Geometry::Point(Point::new(2., 4.));
//! let res = geom_to_wkb(&p);
//! assert_eq!(res, vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 16, 64]);
//! # }
//! ```
//!
//! You can also 'read' a Geometry from a `std::io::Read`:
//!
//! ```rust
//! # extern crate geo;
//! # extern crate wkb;
//! # fn main() {
//! use geo::*;
//! use wkb::*;
//!
//! let bytes: Vec<u8> = vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 16, 64];
//! let p: Geometry<f64> = wkb_to_geom(bytes.as_slice());
//! assert_eq!(p, Geometry::Point(Point::new(2., 4.)));
//! # }
//! ```
//!
//! Adding proper `*Ext` traits is planned.
//!
//!
extern crate geo;
extern crate byteorder;
extern crate num_traits;

use std::io::prelude::*;

use geo::*;
use num_traits::Float;
use byteorder::{WriteBytesExt, ReadBytesExt};
use byteorder::{LittleEndian};

fn read_point<I: Read>(mut wkb: I) -> Point<f64> {
    let x: f64 = wkb.read_f64::<LittleEndian>().unwrap();
    let y: f64 = wkb.read_f64::<LittleEndian>().unwrap();
    Point::new(x, y)
}

fn write_point<W: Write, T: Into<f64>+Float>(p: &Point<T>, out: &mut W) {
    out.write_f64::<LittleEndian>(p.x().into());
    out.write_f64::<LittleEndian>(p.y().into());
}

fn read_many_points<I: Read>(mut wkb: I) -> Vec<Point<f64>> {
    let num_points = wkb.read_u32::<LittleEndian>().unwrap() as usize;
    let mut res: Vec<Point<f64>> = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        res.push(read_point(&mut wkb));
    }

    res
}

fn write_many_points<W: Write, T: Into<f64>+Float>(mp: &[Point<T>], mut out: &mut W) {
    out.write_u32::<LittleEndian>(mp.len() as u32);
    for p in mp.iter() {
        write_point(p, &mut out);
    }
}

/// Convert a Geometry into WKB bytes.
pub fn geom_to_wkb<T: Into<f64>+Float>(geom: &geo::Geometry<T>) -> Vec<u8> {
    let mut result: Vec<u8> = Vec::new();
    write_geom_to_wkb(geom, &mut result);
    result
}

/// Write a geometry to the underlying writer.
pub fn write_geom_to_wkb<W: Write, T: Into<f64>+Float>(geom: &geo::Geometry<T>, mut result: &mut W) {
    // FIXME replace type signature with Into<Geometry<T>>
    
    // little endian
    result.write_u8(1);
    match geom {
        &Geometry::Point(p) => {
            result.write_u32::<LittleEndian>(1);
            write_point(&p, &mut result);
        },
        &Geometry::LineString(ref ls) => {
            result.write_u32::<LittleEndian>(2);
            write_many_points(&ls.0, &mut result);
        },
        &Geometry::Polygon(ref p) => {
            result.write_u32::<LittleEndian>(3);
            result.write_u32::<LittleEndian>(1 + p.interiors.len() as u32);
            write_many_points(&p.exterior.0, &mut result);
            for i in p.interiors.iter() {
                write_many_points(&i.0, &mut result);
            }
        }
        &Geometry::MultiPoint(ref mp) => {
            result.write_u32::<LittleEndian>(4);
            write_many_points(&mp.0, &mut result);
        },
        &Geometry::MultiLineString(ref mls) => {
            result.write_u32::<LittleEndian>(5);
            result.write_u32::<LittleEndian>(mls.0.len() as u32);
            for ls in mls.0.iter() {
                write_many_points(&ls.0, &mut result);
            }
        },
        &Geometry::MultiPolygon(ref mp) => {
            result.write_u32::<LittleEndian>(6);
            result.write_u32::<LittleEndian>(mp.0.len() as u32);
            for poly in mp.0.iter() {
                result.write_u32::<LittleEndian>(1 + poly.interiors.len() as u32);

                write_many_points(&poly.exterior.0, &mut result);
                for int in poly.interiors.iter() {
                    write_many_points(&int.0, &mut result);
                }
            }
        },
        &Geometry::GeometryCollection(ref gc) => {
            // FIXME implement, don't want to duplicate all the above
            unimplemented!();
        }
    }

}

/// Read a Geometry from a reader. Converts WKB to a Geometry.
pub fn wkb_to_geom<I: Read>(mut wkb: I) -> geo::Geometry<f64> {
    match wkb.read_u8().unwrap() {
        0 => unimplemented!(),
        1 => { },  // LittleEndian, OK
        _ => panic!(),
    };

    match wkb.read_u32::<LittleEndian>().unwrap() {
        1 => {
            // Point
            Geometry::Point(read_point(&mut wkb))
        },
        2 => {
            // LineString
            let points = read_many_points(&mut wkb);
            Geometry::LineString(LineString(points))
        },
        3 => {
            // Polygon
            let num_rings = wkb.read_u32::<LittleEndian>().unwrap() as usize;
            let exterior = read_many_points(&mut wkb);
            let mut interiors = Vec::with_capacity(num_rings-1);
            for _ in 0..(num_rings-1) {
                interiors.push(LineString(read_many_points(&mut wkb)));
            }
            Geometry::Polygon(Polygon::new(LineString(exterior), interiors))
        },
        4 => {
            // MultiPoint
            let points = read_many_points(&mut wkb);
            Geometry::MultiPoint(MultiPoint(points))
        },
        5 => {
            // MultiLineString
            let num_linestrings = wkb.read_u32::<LittleEndian>().unwrap() as usize;
            let mut linestrings = Vec::with_capacity(num_linestrings);
            for _ in 0..num_linestrings {
                linestrings.push(LineString(read_many_points(&mut wkb)));
            }
            Geometry::MultiLineString(MultiLineString(linestrings))
        },
        6 => {
            // MultiPolygon
            let num_polygons = wkb.read_u32::<LittleEndian>().unwrap() as usize;
            let mut polygons = Vec::with_capacity(num_polygons);
            for _ in 0..num_polygons {
                let num_rings = wkb.read_u32::<LittleEndian>().unwrap() as usize;
                let exterior = LineString(read_many_points(&mut wkb));
                let mut interiors = Vec::with_capacity(num_rings-1);
                for _ in 0..(num_rings-1) {
                    interiors.push(LineString(read_many_points(&mut wkb)));
                }
                polygons.push(Polygon::new(exterior, interiors));
            }

            Geometry::MultiPolygon(MultiPolygon(polygons))
        },
        _ => unimplemented!(),
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    fn assert_two_f64<R: Read, I: Into<f64>>(mut reader: &mut R, a: I, b: I) {
        assert_eq!(reader.read_f64::<LittleEndian>().unwrap(), a.into());
        assert_eq!(reader.read_f64::<LittleEndian>().unwrap(), b.into());
    }

    fn write_two_f64<W: Write, F: Into<f64>>(mut writer: &mut W, a: F, b: F) {
        writer.write_f64::<LittleEndian>(a.into());
        writer.write_f64::<LittleEndian>(b.into());
    }

    #[test]
    fn point_to_wkb() {
        let p: Geometry<f64> = Geometry::Point(Point::new(2., 4.));
        let res = geom_to_wkb(&p);
        let mut res = res.as_slice();
        assert_eq!(res.read_u8().unwrap(), 1);
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 1);
        assert_two_f64(&mut res, 2, 4);

        assert_eq!(wkb_to_geom(geom_to_wkb(&p).as_slice()), p);
    }

    #[test]
    fn wkb_to_point() {
        let mut bytes = Vec::new();
        bytes.write_u8(1);
        bytes.write_u32::<LittleEndian>(1);
        bytes.write_f64::<LittleEndian>(100.);
        bytes.write_f64::<LittleEndian>(-2.);

        let geom = wkb_to_geom(bytes.as_slice());
        // TODO need a geom.is_point()
        if let Geometry::Point(p) = geom {
            assert_eq!(p.x(), 100.);
            assert_eq!(p.y(), -2.);
        } else {
            assert!(false);
        }

        assert_eq!(geom_to_wkb(&wkb_to_geom(bytes.as_slice())), bytes);
    }

    #[test]
    fn linestring_to_wkb() {
        let mut ls = LineString(vec![]);
        ls.0.push(Point::new(0., 0.));
        ls.0.push(Point::new(1., 0.));
        ls.0.push(Point::new(1., 1.));
        ls.0.push(Point::new(0., 1.));
        ls.0.push(Point::new(0., 0.));
        let ls = Geometry::LineString(ls);

        let res = geom_to_wkb(&ls);
        let mut res = res.as_slice();
        assert_eq!(res.read_u8().unwrap(), 1);  // LittleEndian
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 2);  // 2 = Linestring
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 5);  // num points

        assert_two_f64(&mut res, 0, 0);
        assert_two_f64(&mut res, 1, 0);
        assert_two_f64(&mut res, 1, 1);
        assert_two_f64(&mut res, 0, 1);
        assert_two_f64(&mut res, 0, 0);

        assert_eq!(wkb_to_geom(geom_to_wkb(&ls).as_slice()), ls);
    }

    #[test]
    fn wkb_to_linestring() {
        let mut bytes = Vec::new();
        bytes.write_u8(1);

        bytes.write_u32::<LittleEndian>(2);
        bytes.write_u32::<LittleEndian>(2);

        write_two_f64(&mut bytes, 0, 0);
        write_two_f64(&mut bytes, 1000, 1000);

        let geom = wkb_to_geom(bytes.as_slice());
        if let Geometry::LineString(ls) = geom {
            assert_eq!(ls.0.len(), 2);
            assert_eq!(ls.0[0].x(), 0.);
            assert_eq!(ls.0[0].y(), 0.);
            assert_eq!(ls.0[1].x(), 1000.);
            assert_eq!(ls.0[1].y(), 1000.);
        } else {
            assert!(false);
        }

        assert_eq!(geom_to_wkb(&wkb_to_geom(bytes.as_slice())), bytes);
    }


    #[test]
    fn polygon_to_wkb() {
        let mut ls = LineString(vec![]);
        ls.0.push(Point::new(0., 0.));
        ls.0.push(Point::new(10., 0.));
        ls.0.push(Point::new(10., 10.));
        ls.0.push(Point::new(0., 10.));
        ls.0.push(Point::new(0., 0.));

        let mut int = LineString(vec![]);
        int.0.push(Point::new(2., 2.));
        int.0.push(Point::new(2., 4.));
        int.0.push(Point::new(4., 4.));
        int.0.push(Point::new(4., 2.));
        int.0.push(Point::new(2., 2.));
        let p = Geometry::Polygon(Polygon::new(ls, vec![int]));

        let res = geom_to_wkb(&p);
        let mut res = res.as_slice();
        assert_eq!(res.read_u8().unwrap(), 1);
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 3);
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 2);

        // Exterior Ring
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 5);

        assert_two_f64(&mut res, 0, 0);
        assert_two_f64(&mut res, 10, 0);
        assert_two_f64(&mut res, 10, 10);
        assert_two_f64(&mut res, 0, 10);
        assert_two_f64(&mut res, 0, 0);

        // interior ring
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 5);

        assert_two_f64(&mut res, 2, 2);
        assert_two_f64(&mut res, 2, 4);
        assert_two_f64(&mut res, 4, 4);
        assert_two_f64(&mut res, 4, 2);
        assert_two_f64(&mut res, 2, 2);

        assert_eq!(wkb_to_geom(geom_to_wkb(&p).as_slice()), p);
    }

    #[test]
    fn wkb_to_polygon() {
        let mut bytes = Vec::new();
        bytes.write_u8(1);
        bytes.write_u32::<LittleEndian>(3);
        bytes.write_u32::<LittleEndian>(1);
        bytes.write_u32::<LittleEndian>(3);

        write_two_f64(&mut bytes, 0, 0);
        write_two_f64(&mut bytes, 1, 0);
        write_two_f64(&mut bytes, 0, 1);

        let geom = wkb_to_geom(bytes.as_slice());
        if let Geometry::Polygon(p) = geom {
            assert_eq!(p.interiors.len(), 0);
            assert_eq!(p.exterior.0.len(), 3);
            assert_eq!(p.exterior.0[0], Point::new(0., 0.));
            assert_eq!(p.exterior.0[1], Point::new(1., 0.));
            assert_eq!(p.exterior.0[2], Point::new(0., 1.));
        } else {
            assert!(false);
        }

        assert_eq!(geom_to_wkb(&wkb_to_geom(bytes.as_slice())), bytes);
    }

    #[test]
    fn multipoint_to_wkb() {
        let p: Geometry<f64> = Geometry::MultiPoint(MultiPoint(vec![Point::new(0., 0.), Point::new(10., -2.)]));
        let res = geom_to_wkb(&p);
        let mut res = res.as_slice();
        assert_eq!(res.read_u8().unwrap(), 1);
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 4);
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 2);
        assert_two_f64(&mut res, 0, 0);
        assert_two_f64(&mut res, 10, -2);

        assert_eq!(wkb_to_geom(geom_to_wkb(&p).as_slice()), p);
    }

    #[test]
    fn wkb_to_multipoing() {
        let mut bytes = Vec::new();
        bytes.write_u8(1);
        bytes.write_u32::<LittleEndian>(4);
        bytes.write_u32::<LittleEndian>(1);
        write_two_f64(&mut bytes, 100, -2);

        let geom = wkb_to_geom(bytes.as_slice());
        if let Geometry::MultiPoint(mp) = geom {
            assert_eq!(mp.0.len(), 1);
            assert_eq!(mp.0[0].x(), 100.);
            assert_eq!(mp.0[0].y(), -2.);
        } else {
            assert!(false);
        }

        assert_eq!(geom_to_wkb(&wkb_to_geom(bytes.as_slice())), bytes);
    }

    #[test]
    fn multilinestring_to_wkb() {
        let ls = Geometry::MultiLineString(MultiLineString(vec![
                    LineString(vec![Point::new(0., 0.), Point::new(1., 1.)]),
                    LineString(vec![Point::new(10., 10.), Point::new(10., 11.)]),
                   ]));

        let res = geom_to_wkb(&ls);
        let mut res = res.as_slice();
        assert_eq!(res.read_u8().unwrap(), 1);
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 5);
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 2);

        // Exterior Ring
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 2);
        assert_two_f64(&mut res, 0, 0);
        assert_two_f64(&mut res, 1, 1);

        // interior ring
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 2);
        assert_two_f64(&mut res, 10, 10);
        assert_two_f64(&mut res, 10, 11);

        assert_eq!(wkb_to_geom(geom_to_wkb(&ls).as_slice()), ls);
    }

    #[test]
    fn wkb_to_multilinestring() {
        let mut bytes = Vec::new();
        bytes.write_u8(1);
        bytes.write_u32::<LittleEndian>(5);
        bytes.write_u32::<LittleEndian>(1);

        bytes.write_u32::<LittleEndian>(3);
        write_two_f64(&mut bytes, 0, 0);
        write_two_f64(&mut bytes, 1, 0);
        write_two_f64(&mut bytes, 0, 1);

        let geom = wkb_to_geom(bytes.as_slice());
        if let Geometry::MultiLineString(mls) = geom {
            assert_eq!(mls.0.len(), 1);
            assert_eq!(mls.0[0].0.len(), 3);
            assert_eq!(mls.0[0].0[0], Point::new(0., 0.));
            assert_eq!(mls.0[0].0[1], Point::new(1., 0.));
            assert_eq!(mls.0[0].0[2], Point::new(0., 1.));
        } else {
            assert!(false);
        }

        assert_eq!(geom_to_wkb(&wkb_to_geom(bytes.as_slice())), bytes);
    }


    #[test]
    fn multipolygon_to_wkb() {
        let mut ls = LineString(vec![]);
        ls.0.push(Point::new(0., 0.));
        ls.0.push(Point::new(10., 0.));
        ls.0.push(Point::new(10., 10.));
        ls.0.push(Point::new(0., 10.));
        ls.0.push(Point::new(0., 0.));

        let mut int = LineString(vec![]);
        int.0.push(Point::new(2., 2.));
        int.0.push(Point::new(2., 4.));
        int.0.push(Point::new(2., 2.));
        let p1 = Polygon::new(ls, vec![]);
        let p2 = Polygon::new(int, vec![]);
        let p = Geometry::MultiPolygon(MultiPolygon(vec![p1, p2]));

        let res = geom_to_wkb(&p);
        let mut res = res.as_slice();
        assert_eq!(res.read_u8().unwrap(), 1);
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 6);
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 2);

        // polygon 1
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 1); // only one ring

        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 5); // 5 points in ring #1
        assert_two_f64(&mut res, 0, 0);
        assert_two_f64(&mut res, 10, 0);
        assert_two_f64(&mut res, 10, 10);
        assert_two_f64(&mut res, 0, 10);
        assert_two_f64(&mut res, 0, 0);

        // polygon 2
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 1); // one ring
        assert_eq!(res.read_u32::<LittleEndian>().unwrap(), 3); // 3 points in ring #1

        assert_two_f64(&mut res, 2, 2);
        assert_two_f64(&mut res, 2, 4);
        assert_two_f64(&mut res, 2, 2);

        assert_eq!(wkb_to_geom(geom_to_wkb(&p).as_slice()), p);
    }

    #[test]
    fn wkb_to_multipolygon() {
        let mut bytes = Vec::new();
        bytes.write_u8(1);
        bytes.write_u32::<LittleEndian>(6);
        bytes.write_u32::<LittleEndian>(2);

        // polygon #1
        bytes.write_u32::<LittleEndian>(2); // 2 rings

        // ring #1 (ext ring)
        bytes.write_u32::<LittleEndian>(4);
        write_two_f64(&mut bytes, 0, 0);
        write_two_f64(&mut bytes, 10, 0);
        write_two_f64(&mut bytes, 10, 10);
        write_two_f64(&mut bytes, 0, 0);

        // ring #2 (int ring)
        bytes.write_u32::<LittleEndian>(4);
        write_two_f64(&mut bytes, 1, 1);
        write_two_f64(&mut bytes, 2, 1);
        write_two_f64(&mut bytes, 2, 2);
        write_two_f64(&mut bytes, 1, 1);

        // polygon #2
        bytes.write_u32::<LittleEndian>(1); // 1 ring
        bytes.write_u32::<LittleEndian>(5); // 5 points
        write_two_f64(&mut bytes, 0, 10);
        write_two_f64(&mut bytes, 10, 100);
        write_two_f64(&mut bytes, 11, 100);
        write_two_f64(&mut bytes, 11, 101);
        write_two_f64(&mut bytes, 10, 10);

        let geom = wkb_to_geom(bytes.as_slice());
        if let Geometry::MultiPolygon(mp) = geom {
            assert_eq!(mp.0.len(), 2);
            assert_eq!(mp.0[0].exterior.0.len(), 4);
            assert_eq!(mp.0[0].exterior.0[0], Point::new(0., 0.));
            assert_eq!(mp.0[0].exterior.0[1], Point::new(10., 0.));
            assert_eq!(mp.0[0].exterior.0[2], Point::new(10., 10.));
            assert_eq!(mp.0[0].exterior.0[3], Point::new(0., 0.));

            assert_eq!(mp.0[0].interiors.len(), 1);
            assert_eq!(mp.0[0].interiors[0].0.len(), 4);
            assert_eq!(mp.0[0].interiors[0].0[0], Point::new(1., 1.));
            assert_eq!(mp.0[0].interiors[0].0[1], Point::new(2., 1.));
            assert_eq!(mp.0[0].interiors[0].0[2], Point::new(2., 2.));
            assert_eq!(mp.0[0].interiors[0].0[3], Point::new(1., 1.));

            assert_eq!(mp.0[1].exterior.0.len(), 5);
            assert_eq!(mp.0[1].exterior.0[0], Point::new(0., 10.));
            assert_eq!(mp.0[1].exterior.0[1], Point::new(10., 100.));
            assert_eq!(mp.0[1].exterior.0[2], Point::new(11., 100.));
            assert_eq!(mp.0[1].exterior.0[3], Point::new(11., 101.));
            assert_eq!(mp.0[1].exterior.0[4], Point::new(10., 10.));
            assert_eq!(mp.0[1].interiors.len(), 0);
        } else {
            assert!(false);
        }

    }

    #[allow(dead_code)]
    fn geometrycollection_to_wkb() {
        // FIXME finish
        let p: Geometry<_> = Point::new(0., 0.).into();
        let l: Geometry<_> = LineString(vec![Point::new(10., 0.), Point::new(20., 0.)]).into();
        let gc: Geometry<_> = GeometryCollection(vec![p, l]).into();

        let res = geom_to_wkb(&gc);
        let mut res = res.as_slice();
        assert_eq!(res.read_u8().unwrap(), 1);
    }

}
