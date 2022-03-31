//! This crate provides functions to convert `rust-geo` geometry types to and from Well Known
//! Binary format.
//!
//! # Examples
//!
//! ```rust
//! # extern crate geo_types;
//! # extern crate wkb;
//! # fn main() {
//! use geo_types::*;
//! use wkb::*;
//!
//! let p: Geometry<f64> = Geometry::Point(Point::new(2., 4.));
//! let res = geom_to_wkb(&p).unwrap();
//! assert_eq!(res, vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 16, 64]);
//! # }
//! ```
//!
//! You can also 'read' a Geometry from a `std::io::Read`:
//!
//! ```rust
//! # extern crate geo_types;
//! # extern crate wkb;
//! # fn main() {
//! use std::io::prelude::*;
//! use std::io::Cursor;
//! use geo_types::*;
//! use wkb::*;
//!
//! let bytes: Vec<u8> = vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 16, 64];
//! let mut bytes_cursor = Cursor::new(bytes);
//! let p = bytes_cursor.read_wkb().unwrap();
//! assert_eq!(p, Geometry::Point(Point::new(2., 4.)));
//! # }
//! ```
//!
//! `.write_wkb(Geometry<Into<f64>>)` works similar:
//!
//! ```rust
//! # extern crate geo_types;
//! # extern crate wkb;
//! # fn main() {
//! # use std::io::prelude::*;
//! # use geo_types::*;
//! # use wkb::*;
//!
//! let mut bytes: Vec<u8> = vec![];
//! bytes.write_wkb(&Geometry::Point(Point::new(2_f64, 4.))).unwrap();
//! assert_eq!(bytes, vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 16, 64]);
//! # }
//! ```
//!
//!

extern crate geo_types;
extern crate num_traits;

use std::fmt::Debug;
use std::io;
use std::io::prelude::*;

use geo_types::*;
use num_traits::Float;

const LITTLE_ENDIAN: &[u8] = &[1];

/// Extension trait for `Read`
pub trait WKBReadExt {
    /// Attempt to read a Geometry<f64> from this reader
    fn read_wkb(&mut self) -> Result<Geometry<f64>, WKBReadError>;
}

impl<R: Read> WKBReadExt for R {
    #[inline]
    fn read_wkb(&mut self) -> Result<Geometry<f64>, WKBReadError>
    {
        wkb_to_geom(self)
    }
}


/// Extension trait for `Write`
pub trait WKBWriteExt {
    /// Attempt to write a Geometry<Into<f64>> to this writer.
    fn write_wkb<T>(&mut self, g: &Geometry<T>) -> Result<(), WKBWriteError>
        where
            T: Into<f64>+Float+Debug;
}

impl<W: Write> WKBWriteExt for W {
    fn write_wkb<T>(&mut self, g: &Geometry<T>) -> Result<(), WKBWriteError>
        where
            T: Into<f64>+Float+Debug,
    {
        write_geom_to_wkb(g, self)
    }
}

/// An error occured when reading
#[derive(Debug)]
pub enum WKBReadError {
    /// This WKB is in BigEndian format, which this library does not yet support.
    UnsupportedBigEndian,

    /// Within in the format, there was an unexpected or wrong data type
    WrongType,

    /// Underlying IO error from the Reader
    IOError(io::Error),
}

impl From<io::Error> for WKBReadError {
    fn from(err: io::Error) -> WKBReadError {
        WKBReadError::IOError(err)
    }
}

/// A thing (`Geometry`) that can be written as WKB
///
/// ```rust
/// # extern crate geo_types;
/// # extern crate wkb;
/// # fn main() {
/// use geo_types::*;
/// use wkb::*;
/// let p: Geometry<f64> = Geometry::Point(Point::new(2., 4.));
/// let mut bytes = Vec::new();
/// p.write_as_wkb(&mut bytes).unwrap();
/// assert_eq!(bytes, vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 16, 64]);
///
/// let p2 = Point::new(2., 4.);
/// let mut bytes = Vec::new();
/// p2.write_as_wkb(&mut bytes).unwrap();
/// assert_eq!(bytes, vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 16, 64]);
/// # }
/// ```
pub trait WKBSerializable {
    /// Attempt to write self as WKB to a `Write`.
    fn write_as_wkb(&self, w: &mut (impl Write + ?Sized)) -> Result<(), WKBWriteError>;

    /// Return self as WKB bytes
    fn as_wkb_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.write_as_wkb(&mut bytes).unwrap();
        bytes
    }
}

/// A thing (`Geometry`) that can be read as WKB
///
/// ```rust
/// # extern crate geo_types;
/// # extern crate wkb;
/// # fn main() {
/// use geo_types::*;
/// use wkb::*;
/// let bytes: Vec<u8> = vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 16, 64];
/// let p: Geometry<f64> = Geometry::read_from_wkb(&mut &*bytes).unwrap();
/// assert_eq!(p, Geometry::Point(Point::new(2., 4.)));
///
/// let p2: Point<f64> = Point::read_from_wkb(&mut &*bytes).unwrap();
/// assert_eq!(p2, Point::new(2., 4.));
/// # }
/// ```
pub trait WKBUnserializable {
    /// Attempt to read an instance of self from this `Read`.
    fn read_from_wkb(r: &mut impl Read) -> Result<Self, WKBReadError> where Self: Sized;
}

/// A thing (`Geometry`) that can be read or written as WKB
///
/// ```rust
/// # extern crate geo_types;
/// # extern crate wkb;
/// # fn main() {
/// use geo_types::*;
/// use wkb::*;
/// let p: Geometry<f64> = Geometry::Point(Point::new(2., 4.));
/// let mut bytes = Vec::new();
/// p.write_as_wkb(&mut bytes).unwrap();
/// assert_eq!(bytes, vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 16, 64]);
///
/// //let p2 = Point::new(2., 4.);
/// //let mut bytes = Vec::new();
/// //p2.write_as_wkb(&mut bytes).unwrap();
/// //assert_eq!(bytes, vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 16, 64]);
/// # }
/// ```
pub trait WKBAbleExt: WKBSerializable + WKBUnserializable {}

impl<T> WKBAbleExt for T where T: ?Sized + WKBSerializable + WKBUnserializable {}

impl<T> WKBSerializable for Geometry<T> where T: Into<f64> + Float + Debug
{
    fn write_as_wkb(&self, w: &mut (impl Write + ?Sized)) -> Result<(), WKBWriteError>
    {
        write_geom_to_wkb(self, w)
    }
}

impl WKBUnserializable for Geometry<f64>
{
    fn read_from_wkb(r: &mut impl Read) -> Result<Self, WKBReadError>
    {
        wkb_to_geom(r)
    }
}

impl<T> WKBSerializable for Point<T> where T: Into<f64> + Float + Debug
{
    #[inline]
    fn write_as_wkb(&self, w: &mut (impl Write + ?Sized)) -> Result<(), WKBWriteError>
    {
        w.write(LITTLE_ENDIAN)?;
        w.write_all(&1_u32.to_le_bytes())?;
        write_point(&self.0, w)
    }
}

impl WKBUnserializable for Point<f64> {
    fn read_from_wkb(r: &mut impl Read) -> Result<Self, WKBReadError>
    {
        match wkb_to_geom(r)? {
            Geometry::Point(p) => Ok(p),
            _ => Err(WKBReadError::WrongType)
        }
    }
}

impl<T> WKBSerializable for LineString<T> where T: Into<f64> + Float + Debug
{
    #[inline]
    fn write_as_wkb(&self, w: &mut (impl Write + ?Sized)) -> Result<(), WKBWriteError>
    {
        w.write(LITTLE_ENDIAN)?;
        w.write_all(&2_u32.to_le_bytes())?;
        write_many_points(&self.0, w)
    }
}

impl WKBUnserializable for LineString<f64> {
    fn read_from_wkb(r: &mut impl Read) -> Result<Self, WKBReadError>
    {
        match wkb_to_geom(r)? {
            Geometry::LineString(l) => Ok(l),
            _ => Err(WKBReadError::WrongType)
        }
    }
}

impl<T> WKBSerializable for Polygon<T> where T: Into<f64> + Float + Debug
{
    fn write_as_wkb(&self, w: &mut (impl Write + ?Sized)) -> Result<(), WKBWriteError>
    {
        w.write(LITTLE_ENDIAN)?;
        w.write_all(&(3_u32).to_le_bytes())?;
        w.write_all(&(1 + self.interiors().len() as u32).to_le_bytes())?;
        write_many_points(&self.exterior().0, w)?;
        for i in self.interiors().iter() {
            write_many_points(&i.0, w)?;
        }
        Ok(())
    }
}

impl WKBUnserializable for Polygon<f64> {
    fn read_from_wkb(r: &mut impl Read) -> Result<Self, WKBReadError>
    {
        match wkb_to_geom(r)? {
            Geometry::Polygon(p) => Ok(p),
            _ => Err(WKBReadError::WrongType)
        }
    }
}

impl<T> WKBSerializable for MultiPoint<T> where T: Into<f64> + Float + Debug
{
    fn write_as_wkb(&self, w: &mut (impl Write + ?Sized)) -> Result<(), WKBWriteError>
    {
        w.write(LITTLE_ENDIAN)?;
        w.write_all(&(4_u32).to_le_bytes())?;
        write_many_points(
            &self.0.iter().map(|p| p.0).collect::<Vec<Coordinate<T>>>(),
            w,
        )
    }
}

impl WKBUnserializable for MultiPoint<f64> {
    fn read_from_wkb(r: &mut impl Read) -> Result<Self, WKBReadError>
    {
        match wkb_to_geom(r)? {
            Geometry::MultiPoint(mp) => Ok(mp),
            _ => Err(WKBReadError::WrongType)
        }
    }
}

impl<T> WKBSerializable for MultiLineString<T> where T: Into<f64> + Float + Debug
{
    fn write_as_wkb(&self, w: &mut (impl Write + ?Sized)) -> Result<(), WKBWriteError>
    {
        w.write(LITTLE_ENDIAN)?;
        w.write_all(&(5_u32).to_le_bytes())?;
        w.write_all(&(self.0.len() as u32).to_le_bytes())?;
        for ls in self.0.iter() {
            ls.write_as_wkb(w)?
        }
        Ok(())
    }
}

impl WKBUnserializable for MultiLineString<f64> {
    fn read_from_wkb(r: &mut impl Read) -> Result<Self, WKBReadError>
    {
        match wkb_to_geom(r)? {
            Geometry::MultiLineString(ml) => Ok(ml),
            _ => Err(WKBReadError::WrongType)
        }
    }
}

impl<T> WKBSerializable for MultiPolygon<T> where T: Into<f64> + Float + Debug
{
    fn write_as_wkb(&self, w: &mut (impl Write + ?Sized)) -> Result<(), WKBWriteError>
    {
        w.write(LITTLE_ENDIAN)?;
        w.write_all(&(6_u32).to_le_bytes())?;
        w.write_all(&(self.0.len() as u32).to_le_bytes())?;
        for poly in self.0.iter() {
            poly.write_as_wkb(w)?
        }
        Ok(())
    }
}

impl WKBUnserializable for MultiPolygon<f64> {
    fn read_from_wkb(r: &mut impl Read) -> Result<Self, WKBReadError>
    {
        match wkb_to_geom(r)? {
            Geometry::MultiPolygon(mp) => Ok(mp),
            _ => Err(WKBReadError::WrongType)
        }
    }
}

impl<T> WKBSerializable for GeometryCollection<T> where T: Into<f64> + Float + Debug
{
    fn write_as_wkb(&self, w: &mut (impl Write + ?Sized)) -> Result<(), WKBWriteError>
    {
        w.write(LITTLE_ENDIAN)?;
        w.write_all(&(7_u32).to_le_bytes())?;
        w.write_all(&(self.len() as u32).to_le_bytes())?;
        for geom in self.0.iter() {
            write_geom_to_wkb(geom, w)?;
        }
        Ok(())
    }
}

impl WKBUnserializable for GeometryCollection<f64> {
    fn read_from_wkb(r: &mut impl Read) -> Result<Self, WKBReadError>
    {
        match wkb_to_geom(r)? {
            Geometry::GeometryCollection(gc) => Ok(gc),
            _ => Err(WKBReadError::WrongType)
        }
    }
}

/// An error occured when writing
#[derive(Debug)]
pub enum WKBWriteError {
    /// Geometry is a `geo_types::Rect`, which this library does not yet support
    UnsupportedGeoTypeRect,

    /// Geometry is a `geo_types::Triangle`, which this library does not yet support
    UnsupportedGeoTypeTriangle,

    /// An IO Error
    IOError(io::Error),
}

impl From<io::Error> for WKBWriteError {
    fn from(err: io::Error) -> WKBWriteError {
        WKBWriteError::IOError(err)
    }
}

fn read_f64(mut rdr: impl Read) -> Result<f64, std::io::Error> {
    let mut bytes = [0; 8];
    rdr.read_exact(&mut bytes)?;

    Ok(f64::from_le_bytes(bytes))
}

fn read_u32(mut rdr: impl Read) -> Result<u32, std::io::Error> {
    let mut bytes = [0; 4];
    rdr.read_exact(&mut bytes)?;

    Ok(u32::from_le_bytes(bytes))
}

fn read_u8(mut rdr: impl Read) -> Result<u8, std::io::Error> {
    let mut bytes = [0; 1];
    rdr.read_exact(&mut bytes)?;

    Ok(bytes[0])
}

fn read_point(mut wkb: impl Read) -> Result<Coordinate<f64>, WKBReadError> {
    let x: f64 = read_f64(&mut wkb)?;
    let y: f64 = read_f64(&mut wkb)?;
    Ok(Coordinate { x, y })
}

fn write_point<W: Write + ?Sized, T: Into<f64> + Float + Debug>(
    c: &Coordinate<T>,
    out: &mut W,
) -> Result<(), WKBWriteError> {
    out.write_all(&c.x.into().to_le_bytes())?;
    out.write_all(&c.y.into().to_le_bytes())?;
    Ok(())
}

fn read_many_points<I: Read>(mut wkb: I) -> Result<Vec<Coordinate<f64>>, WKBReadError> {
    let num_points = read_u32(&mut wkb)? as usize;
    let mut res: Vec<Coordinate<f64>> = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        res.push(read_point(&mut wkb)?);
    }

    Ok(res)
}

fn write_many_points<W: Write + ?Sized, T: Into<f64> + Float + Debug>(
    mp: &[Coordinate<T>],
    mut out: &mut W,
) -> Result<(), WKBWriteError> {
    out.write_all(&(mp.len() as u32).to_le_bytes())?;
    for p in mp.iter() {
        write_point(p, &mut out)?;
    }

    Ok(())
}

/// Convert a Geometry into WKB bytes.
pub fn geom_to_wkb<T: Into<f64> + Float + Debug>(geom: &Geometry<T>) -> Result<Vec<u8>, WKBWriteError> {
    let mut result: Vec<u8> = Vec::new();
    write_geom_to_wkb(geom, &mut result)?;
    Ok(result)
}

/// Write a geometry to the underlying writer, except for the endianity byte.
pub fn write_geom_to_wkb<W, T>(
    geom: &Geometry<T>,
    mut result: &mut W,
) -> Result<(), WKBWriteError>
    where T: Into<f64>+Float+Debug,
          W: Write + ?Sized,
{
    // FIXME replace type signature with Into<Geometry<T>>
    result.write(LITTLE_ENDIAN)?;
    match geom {
        &Geometry::Point(p) => {
            result.write_all(&1_u32.to_le_bytes())?;
            write_point(&p.0, &mut result)?;
        }
        &Geometry::LineString(ref ls) => {
            result.write_all(&2_u32.to_le_bytes())?;
            write_many_points(&ls.0, &mut result)?;
        }
        &Geometry::Line(ref l) => {
            write_many_points(&[l.start, l.end], &mut result)?;
        }
        &Geometry::Polygon(ref p) => {
            result.write_all(&(3_u32).to_le_bytes())?;
            result.write_all(&(1 + p.interiors().len() as u32).to_le_bytes())?;
            write_many_points(&p.exterior().0, &mut result)?;
            for i in p.interiors().iter() {
                write_many_points(&i.0, &mut result)?;
            }
        }
        &Geometry::MultiPoint(ref mp) => {
            result.write_all(&(4_u32).to_le_bytes())?;
            write_many_points(
                &mp.0.iter().map(|p| p.0).collect::<Vec<Coordinate<T>>>(),
                &mut result,
            )?;
        }
        &Geometry::MultiLineString(ref mls) => {
            result.write_all(&(5_u32).to_le_bytes())?;
            result.write_all(&(mls.0.len() as u32).to_le_bytes())?;
            for ls in mls.0.iter() {
                // I tried to have this call write_geom_to_wkb again, but I couldn't get the types
                // working.
                result.write(LITTLE_ENDIAN)?;
                result.write_all(&(2_u32).to_le_bytes())?;
                write_many_points(&ls.0, &mut result)?;
            }
        }
        &Geometry::MultiPolygon(ref mp) => {
            result.write_all(&(6_u32).to_le_bytes())?;
            result.write_all(&(mp.0.len() as u32).to_le_bytes())?;
            for poly in mp.0.iter() {
                result.write(LITTLE_ENDIAN)?;
                result.write_all(&(3_u32).to_le_bytes())?;
                result.write_all(&(1 + poly.interiors().len() as u32).to_le_bytes())?;

                write_many_points(&poly.exterior().0, &mut result)?;
                for int in poly.interiors().iter() {
                    write_many_points(&int.0, &mut result)?;
                }
            }
        }
        &Geometry::GeometryCollection(ref gc) => {
            result.write_all(&(7_u32).to_le_bytes())?;
            result.write_all(&(gc.len() as u32).to_le_bytes())?;
            for geom in gc.0.iter() {
                write_geom_to_wkb(geom, result)?;
            }
        }
        &Geometry::Rect(ref _rect) => {
            return Err(WKBWriteError::UnsupportedGeoTypeRect);
        }
        &Geometry::Triangle(ref _t) => {
            return Err(WKBWriteError::UnsupportedGeoTypeTriangle);
        }
    }

    Ok(())
}

/// Read a Geometry from a reader. Converts WKB to a Geometry.
pub fn wkb_to_geom<R>(mut wkb: &mut R) -> Result<Geometry<f64>, WKBReadError>
    where R: Read + ?Sized
{
    match read_u8(&mut wkb)? {
        0 => {
            return Err(WKBReadError::UnsupportedBigEndian);
        }
        1 => {} // LittleEndian, OK
        _ => panic!(),
    };

    match read_u32(&mut wkb)? {
        1 => {
            // Point
            Ok(Geometry::Point(Point(read_point(&mut wkb)?)))
        }
        2 => {
            // LineString
            let points = read_many_points(&mut wkb)?;
            Ok(Geometry::LineString(LineString(points)))
        }
        3 => {
            // Polygon
            let num_rings = read_u32(&mut wkb)? as usize;
            let exterior = read_many_points(&mut wkb)?;
            let mut interiors = Vec::with_capacity(num_rings - 1);
            for _ in 0..(num_rings - 1) {
                interiors.push(LineString(read_many_points(&mut wkb)?));
            }
            Ok(Geometry::Polygon(Polygon::new(
                LineString(exterior),
                interiors,
            )))
        }
        4 => {
            // MultiPoint
            let points = read_many_points(&mut wkb)?;
            Ok(Geometry::MultiPoint(MultiPoint(
                points.into_iter().map(Point).collect::<Vec<Point<f64>>>(),
            )))
        }
        5 => {
            // MultiLineString
            let num_linestrings = read_u32(&mut wkb)? as usize;
            let mut linestrings = Vec::with_capacity(num_linestrings);
            for _ in 0..num_linestrings {
                let linestring: LineString<f64> = match wkb_to_geom(wkb)? {
                    Geometry::LineString(l) => l,
                    _ => {
                        return Err(WKBReadError::WrongType);
                    }
                };

                linestrings.push(linestring);
            }
            Ok(Geometry::MultiLineString(MultiLineString(linestrings)))
        }
        6 => {
            // MultiPolygon
            let num_polygons = read_u32(&mut wkb)? as usize;
            let mut polygons = Vec::with_capacity(num_polygons);
            for _ in 0..num_polygons {
                let polygon: Polygon<f64> = match wkb_to_geom(wkb)? {
                    Geometry::Polygon(p) => p,
                    _ => {
                        return Err(WKBReadError::WrongType);
                    }
                };

                polygons.push(polygon);
            }

            Ok(Geometry::MultiPolygon(MultiPolygon(polygons)))
        },
        7 => {
            let num_geometries = read_u32(&mut wkb)? as usize;
            let mut geometries = Vec::with_capacity(num_geometries);
            for _i in 0..num_geometries {
                geometries.push(wkb_to_geom(wkb)?);
            }
            Ok(Geometry::GeometryCollection(GeometryCollection(geometries)))
        }
        _ => unimplemented!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_two_f64(mut reader: &mut impl Read, a: impl Into<f64>, b: impl Into<f64>) {
        assert_eq!(read_f64(&mut reader).unwrap(), a.into());
        assert_eq!(read_f64(&mut reader).unwrap(), b.into());
    }

    fn write_two_f64<W: Write>(writer: &mut W, a: impl Into<f64>, b: impl Into<f64>) {
        writer.write_all(&a.into().to_le_bytes()).unwrap();
        writer.write_all(&b.into().to_le_bytes()).unwrap();
    }

    #[test]
    fn point_to_wkb() {
        let p: Geometry<f64> = Geometry::Point(Point::new(2., 4.));
        let res = geom_to_wkb(&p).unwrap();
        let mut res = res.as_slice();
        assert_eq!(read_u8(&mut res).unwrap(), 1);
        assert_eq!(read_u32(&mut res).unwrap(), 1);
        assert_two_f64(&mut res, 2, 4);

        assert_eq!(
            wkb_to_geom(&mut geom_to_wkb(&p).unwrap().as_slice()).unwrap(),
            p
        );
    }

    #[test]
    fn wkb_to_point() {
        let mut bytes = Vec::new();
        bytes.write(LITTLE_ENDIAN).unwrap();
        bytes.write_all(&(1_u32).to_le_bytes()).unwrap();
        bytes.write_all(&(100f64).to_le_bytes()).unwrap();
        bytes.write_all(&(-2f64).to_le_bytes()).unwrap();

        let geom = wkb_to_geom(&mut bytes.as_slice()).unwrap();
        // TODO need a geom.is_point()
        if let Geometry::Point(p) = geom {
            assert_eq!(p.x(), 100.);
            assert_eq!(p.y(), -2.);
        } else {
            assert!(false);
        }

        assert_eq!(
            geom_to_wkb(&wkb_to_geom(&mut bytes.as_slice()).unwrap()).unwrap(),
            bytes
        );
    }

    #[test]
    fn linestring_to_wkb() {
        let ls: LineString<f64> = vec![(0., 0.), (1., 0.), (1., 1.), (0., 1.), (0., 0.)].into();
        let ls = Geometry::LineString(ls);

        let res = geom_to_wkb(&ls).unwrap();
        let mut res = res.as_slice();
        assert_eq!(read_u8(&mut res).unwrap(), 1); // LittleEndian
        assert_eq!(read_u32(&mut res).unwrap(), 2); // 2 = Linestring
        assert_eq!(read_u32(&mut res).unwrap(), 5); // num points

        assert_two_f64(&mut res, 0, 0);
        assert_two_f64(&mut res, 1, 0);
        assert_two_f64(&mut res, 1, 1);
        assert_two_f64(&mut res, 0, 1);
        assert_two_f64(&mut res, 0, 0);

        assert_eq!(
            wkb_to_geom(&mut geom_to_wkb(&ls).unwrap().as_slice()).unwrap(),
            ls
        );
    }

    #[test]
    fn wkb_to_linestring() {
        let mut bytes = Vec::new();
        bytes.write(LITTLE_ENDIAN).unwrap();

        bytes.write_all(&(2_u32).to_le_bytes()).unwrap();
        bytes.write_all(&(2_u32).to_le_bytes()).unwrap();

        write_two_f64(&mut bytes, 0, 0);
        write_two_f64(&mut bytes, 1000, 1000);

        let geom = wkb_to_geom(&mut bytes.as_slice()).unwrap();
        if let Geometry::LineString(ls) = geom {
            assert_eq!(ls.0.len(), 2);
            assert_eq!(ls.0[0].x, 0.);
            assert_eq!(ls.0[0].y, 0.);
            assert_eq!(ls.0[1].x, 1000.);
            assert_eq!(ls.0[1].y, 1000.);
        } else {
            assert!(false);
        }

        assert_eq!(
            geom_to_wkb(&wkb_to_geom(&mut bytes.as_slice()).unwrap()).unwrap(),
            bytes
        );
    }

    #[test]
    fn polygon_to_wkb() {
        let ls = vec![(0., 0.), (10., 0.), (10., 10.), (0., 10.), (0., 0.)].into();
        let int = vec![(2., 2.), (2., 4.), (4., 4.), (4., 2.), (2., 2.)].into();
        let p = Geometry::Polygon(Polygon::new(ls, vec![int]));

        let res = geom_to_wkb(&p).unwrap();
        let mut res = res.as_slice();
        assert_eq!(read_u8(&mut res).unwrap(), 1);
        assert_eq!(read_u32(&mut res).unwrap(), 3);
        assert_eq!(read_u32(&mut res).unwrap(), 2);

        // Exterior Ring
        assert_eq!(read_u32(&mut res).unwrap(), 5);

        assert_two_f64(&mut res, 0, 0);
        assert_two_f64(&mut res, 10, 0);
        assert_two_f64(&mut res, 10, 10);
        assert_two_f64(&mut res, 0, 10);
        assert_two_f64(&mut res, 0, 0);

        // interior ring
        assert_eq!(read_u32(&mut res).unwrap(), 5);

        assert_two_f64(&mut res, 2, 2);
        assert_two_f64(&mut res, 2, 4);
        assert_two_f64(&mut res, 4, 4);
        assert_two_f64(&mut res, 4, 2);
        assert_two_f64(&mut res, 2, 2);

        assert_eq!(
            wkb_to_geom(&mut geom_to_wkb(&p).unwrap().as_slice()).unwrap(),
            p
        );
    }

    #[test]
    fn wkb_to_polygon() {
        let mut bytes = Vec::new();
        bytes.write(LITTLE_ENDIAN).unwrap();
        bytes.write_all(&(3_u32).to_le_bytes()).unwrap();
        bytes.write_all(&(1_u32).to_le_bytes()).unwrap();
        bytes.write_all(&(4_u32).to_le_bytes()).unwrap();

        write_two_f64(&mut bytes, 0, 0);
        write_two_f64(&mut bytes, 1, 0);
        write_two_f64(&mut bytes, 0, 1);
        // WKB requires that polygons are closed
        write_two_f64(&mut bytes, 0, 0);

        let geom = wkb_to_geom(&mut bytes.as_slice()).unwrap();
        if let Geometry::Polygon(p) = geom {
            assert_eq!(p.interiors().len(), 0);
            assert_eq!(p.exterior().0.len(), 4);
            assert_eq!(p.exterior().0[0], (0., 0.).into());
            assert_eq!(p.exterior().0[1], (1., 0.).into());
            assert_eq!(p.exterior().0[2], (0., 1.).into());
            assert_eq!(p.exterior().0[3], (0., 0.).into());
        } else {
            assert!(false);
        }

        assert_eq!(
            geom_to_wkb(&wkb_to_geom(&mut bytes.as_slice()).unwrap()).unwrap(),
            bytes
        );
    }

    #[test]
    fn wkb_to_polygon_auto_closed() {
        let mut bytes = Vec::new();
        bytes.write(LITTLE_ENDIAN).unwrap();
        bytes.write_all(&(3_u32).to_le_bytes()).unwrap();
        bytes.write_all(&(1_u32).to_le_bytes()).unwrap();

        // only 3 points
        bytes.write_all(&(3_u32).to_le_bytes()).unwrap();

        write_two_f64(&mut bytes, 0, 0);
        write_two_f64(&mut bytes, 1, 0);
        write_two_f64(&mut bytes, 0, 1);

        // We /should/ add this
        //write_two_f64(&mut bytes, 0, 0);

        let geom = wkb_to_geom(&mut bytes.as_slice()).unwrap();
        if let Geometry::Polygon(p) = geom {
            assert_eq!(p.interiors().len(), 0);
            assert_eq!(p.exterior().0[0], (0., 0.).into());
            assert_eq!(p.exterior().0[1], (1., 0.).into());
            assert_eq!(p.exterior().0[2], (0., 1.).into());

            // And yet, the geo-types library has add this point
            assert_eq!(p.exterior().0[3], (0., 0.).into());
            assert_eq!(p.exterior().0.len(), 4);
        } else {
            assert!(false);
        }

        // They won't equal
        assert_ne!(
            geom_to_wkb(&wkb_to_geom(&mut bytes.as_slice()).unwrap()).unwrap(),
            bytes
        );
    }

    #[test]
    fn multipoint_to_wkb() {
        let p: Geometry<f64> =
            Geometry::MultiPoint(MultiPoint(vec![Point::new(0., 0.), Point::new(10., -2.)]));
        let res = geom_to_wkb(&p).unwrap();
        let mut res = res.as_slice();
        assert_eq!(read_u8(&mut res).unwrap(), 1);
        assert_eq!(read_u32(&mut res).unwrap(), 4);
        assert_eq!(read_u32(&mut res).unwrap(), 2);
        assert_two_f64(&mut res, 0, 0);
        assert_two_f64(&mut res, 10, -2);

        assert_eq!(
            wkb_to_geom(&mut geom_to_wkb(&p).unwrap().as_slice()).unwrap(),
            p
        );
    }

    #[test]
    fn wkb_to_multipoint() {
        let mut bytes = Vec::new();
        bytes.write(LITTLE_ENDIAN).unwrap();
        bytes.write_all(&(4_u32).to_le_bytes()).unwrap();
        bytes.write_all(&(1_u32).to_le_bytes()).unwrap();
        write_two_f64(&mut bytes, 100, -2);

        let geom = wkb_to_geom(&mut bytes.as_slice()).unwrap();
        if let Geometry::MultiPoint(mp) = geom {
            assert_eq!(mp.0.len(), 1);
            assert_eq!(mp.0[0].x(), 100.);
            assert_eq!(mp.0[0].y(), -2.);
        } else {
            assert!(false);
        }

        assert_eq!(
            geom_to_wkb(&wkb_to_geom(&mut bytes.as_slice()).unwrap()).unwrap(),
            bytes
        );
    }

    #[test]
    fn multilinestring_to_wkb() {
        let ls = Geometry::MultiLineString(MultiLineString(vec![
            vec![(0., 0.), (1., 1.)].into(),
            vec![(10., 10.), (10., 11.)].into(),
        ]));

        let res = geom_to_wkb(&ls).unwrap();
        let mut res = res.as_slice();
        assert_eq!(read_u8(&mut res).unwrap(), 1);
        assert_eq!(read_u32(&mut res).unwrap(), 5); // 5 - MultiLineString
        assert_eq!(read_u32(&mut res).unwrap(), 2); // 2 linestrings

        assert_eq!(read_u8(&mut res).unwrap(), 1);
        assert_eq!(read_u32(&mut res).unwrap(), 2); // 2 = Linestring
        assert_eq!(read_u32(&mut res).unwrap(), 2); // num points
        assert_two_f64(&mut res, 0, 0);
        assert_two_f64(&mut res, 1, 1);

        assert_eq!(read_u8(&mut res).unwrap(), 1);
        assert_eq!(read_u32(&mut res).unwrap(), 2); // 2 = Linestring
        assert_eq!(read_u32(&mut res).unwrap(), 2); // num points
        assert_two_f64(&mut res, 10, 10);
        assert_two_f64(&mut res, 10, 11);

        assert_eq!(
            wkb_to_geom(&mut geom_to_wkb(&ls).unwrap().as_slice()).unwrap(),
            ls
        );
    }

    #[test]
    fn wkb_to_multilinestring() {
        let mut bytes = Vec::new();
        bytes.write(LITTLE_ENDIAN).unwrap();
        bytes.write_all(&(5_u32).to_le_bytes()).unwrap();
        bytes.write_all(&(1_u32).to_le_bytes()).unwrap();

        bytes.write(LITTLE_ENDIAN).unwrap();
        bytes.write_all(&(2_u32).to_le_bytes()).unwrap();
        bytes.write_all(&(3_u32).to_le_bytes()).unwrap();
        write_two_f64(&mut bytes, 0, 0);
        write_two_f64(&mut bytes, 1, 0);
        write_two_f64(&mut bytes, 0, 1);

        let geom = wkb_to_geom(&mut bytes.as_slice()).unwrap();
        if let Geometry::MultiLineString(mls) = geom {
            assert_eq!(mls.0.len(), 1);
            assert_eq!(mls.0[0].0.len(), 3);
            assert_eq!(mls.0[0].0[0].x_y(), (0., 0.));
            assert_eq!(mls.0[0].0[1].x_y(), (1., 0.));
            assert_eq!(mls.0[0].0[2].x_y(), (0., 1.));
        } else {
            assert!(false);
        }

        assert_eq!(
            geom_to_wkb(&wkb_to_geom(&mut bytes.as_slice()).unwrap()).unwrap(),
            bytes
        );
    }

    #[test]
    fn multipolygon_to_wkb() {
        let ls = vec![(0., 0.), (10., 0.), (10., 10.), (0., 10.), (0., 0.)].into();
        let int = vec![(2., 2.), (2., 4.), (2., 2.)].into();

        let p1 = Polygon::new(ls, vec![]);
        let p2 = Polygon::new(int, vec![]);
        let p = Geometry::MultiPolygon(MultiPolygon(vec![p1, p2]));

        let res = geom_to_wkb(&p).unwrap();
        let mut res = res.as_slice();
        assert_eq!(read_u8(&mut res).unwrap(), 1);
        assert_eq!(read_u32(&mut res).unwrap(), 6); // Multipolygon
        assert_eq!(read_u32(&mut res).unwrap(), 2); // num polygons

        // polygon 1
        assert_eq!(read_u8(&mut res).unwrap(), 1);
        assert_eq!(read_u32(&mut res).unwrap(), 3); // polygon
        assert_eq!(read_u32(&mut res).unwrap(), 1); // only one ring

        assert_eq!(read_u32(&mut res).unwrap(), 5); // 5 points in ring #1
        assert_two_f64(&mut res, 0, 0);
        assert_two_f64(&mut res, 10, 0);
        assert_two_f64(&mut res, 10, 10);
        assert_two_f64(&mut res, 0, 10);
        assert_two_f64(&mut res, 0, 0);

        // polygon 2
        assert_eq!(read_u8(&mut res).unwrap(), 1);
        assert_eq!(read_u32(&mut res).unwrap(), 3); // polygon
        assert_eq!(read_u32(&mut res).unwrap(), 1); // one ring
        assert_eq!(read_u32(&mut res).unwrap(), 3); // 3 points in ring #1

        assert_two_f64(&mut res, 2, 2);
        assert_two_f64(&mut res, 2, 4);
        assert_two_f64(&mut res, 2, 2);

        assert_eq!(
            wkb_to_geom(&mut geom_to_wkb(&p).unwrap().as_slice()).unwrap(),
            p
        );
    }

    #[test]
    fn postgis_wkb_to_multipolygon() {
        let bytes: Vec<u8> = vec![
            0x01, 0x06, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x03, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10,
            0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
        ];

        let geom = wkb_to_geom(&mut bytes.as_slice()).unwrap();

        if let Geometry::MultiPolygon(mp) = geom {
            assert_eq!(mp.0.len(), 2);
            assert_eq!(mp.0[0].exterior().0.len(), 5);
            assert_eq!(mp.0[0].exterior().0[0].x_y(), (0., 0.));
            assert_eq!(mp.0[0].exterior().0[1].x_y(), (10., 0.));
            assert_eq!(mp.0[0].exterior().0[2].x_y(), (10., 10.));
            assert_eq!(mp.0[0].exterior().0[3].x_y(), (0., 10.));
            assert_eq!(mp.0[0].exterior().0[4].x_y(), (0., 0.));

            assert_eq!(mp.0[0].interiors().len(), 0);

            assert_eq!(mp.0[1].exterior().0.len(), 4);
            assert_eq!(mp.0[1].exterior().0[0].x_y(), (2., 2.));
            assert_eq!(mp.0[1].exterior().0[1].x_y(), (2., 4.));
            assert_eq!(mp.0[1].exterior().0[2].x_y(), (4., 2.));
            assert_eq!(mp.0[1].exterior().0[0].x_y(), (2., 2.));
            assert_eq!(mp.0[1].interiors().len(), 0);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn geometrycollection_to_wkb() {
        let p: Geometry<_> = Point::new(0., 0.).into();
        let l: Geometry<_> = LineString(vec![(10., 0.).into(), (20., 0.).into()]).into();
        let gc: Geometry<_> = Geometry::GeometryCollection(GeometryCollection(vec![p, l]));

        let res = geom_to_wkb(&gc).unwrap();
        let mut res = res.as_slice();
        assert_eq!(read_u8(&mut res).unwrap(), 1);
        assert_eq!(read_u32(&mut res).unwrap(), 7); // geometry collection
        assert_eq!(read_u32(&mut res).unwrap(), 2);
        assert_eq!(read_u8(&mut res).unwrap(), 1);
        assert_eq!(read_u32(&mut res).unwrap(), 1); // point
        assert_two_f64(&mut res, 0, 0);
        assert_eq!(read_u8(&mut res).unwrap(), 1);
        assert_eq!(read_u32(&mut res).unwrap(), 2);
        assert_eq!(read_u32(&mut res).unwrap(), 2);
        assert_two_f64(&mut res, 10, 0);
        assert_two_f64(&mut res, 20, 0);
    }

    #[test]
    fn test_simple_multilinestring1() {
        let wkb: Vec<u8> = vec![
            1, 5, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 240, 63,
        ];

        let geom = wkb_to_geom(&mut wkb.as_slice()).unwrap();
        assert_eq!(
            geom,
            Geometry::MultiLineString(MultiLineString(vec![vec![(0., 0.), (1., 1.)].into(),]))
        );
    }

    #[test]
    fn test_simple_multilinestring2() {
        let wkb: Vec<u8> = vec![
            0x01, 0x05, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x01, 0x02, 0x00, 0x00, 0x00, 0x02,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
        ];

        let geom = wkb_to_geom(&mut wkb.as_slice()).unwrap();
        assert_eq!(
            geom,
            Geometry::MultiLineString(MultiLineString(vec![
                vec![(0., 0.), (1., 1.)].into(),
                vec![(2., 2.), (3., 2.)].into(),
            ]))
        );
    }

    #[test]
    fn bigendian_not_supported() {
        let mut bytes = Vec::new();
        bytes.write(&[0]).unwrap();
        bytes.write_all(&1_u32.to_be_bytes()).unwrap();
        let res = wkb_to_geom(&mut bytes.as_slice());
        assert!(res.is_err());
    }

    #[test]
    fn wkb_to_geometrycollection() {
        let original_collection = GeometryCollection(vec![Geometry::Point(Point::new(1.5, 2.5)), Geometry::LineString(line_string![(x: 0.5, y: 0.5), (x: 5.5, y: 5.5)])]);
        let original_geometry = Geometry::GeometryCollection(original_collection);
        let wkb = geom_to_wkb(&original_geometry).unwrap();
        let parsed_geometry = wkb_to_geom(&mut wkb.as_slice()).unwrap();
        assert_eq!(original_geometry, parsed_geometry);
    }
}
