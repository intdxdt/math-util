pub use std::f64::consts::{E, SQRT_2, LN_2, PI, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4};
use robust_determinant::{det2 as rob_det2, det3 as rob_det3};

pub const PRECISION: i32 = 12;
pub const EPSILON: f64 = 1.0e-12;
///Indexing X[0], Y[1], Z[2]
const X: usize = 0;
const Y: usize = 1;
const Z: usize = 2;

///Compare two floating point values
#[inline]
pub fn feq_eps(a: f64, b: f64, eps: f64) -> bool {
    (a == b) || ((a - b).abs() < eps)
}

///Compare two floating point values
#[inline]
pub fn feq(a: f64, b: f64) -> bool {
    feq_eps(a, b, EPSILON)
}

///Rounds a float to the nearest whole number float
#[inline]
pub fn round_floor(f: f64) -> f64 {
    return (f + 0.5f64.copysign(f)).trunc();
}

///Rounds a number to the nearest decimal place
#[inline]
pub fn round(x: f64, digits: i32) -> f64 {
    let m = 10f64.powi(digits);
    return round_floor(x * m) / m;
}

#[inline]
pub fn round_0(x: f64) -> f64 {
    round(x, 0)
}

#[inline]
pub fn det2(mat2x2: &[Vec<f64>]) -> f64 {
    *rob_det2(mat2x2).last().unwrap()
}

#[inline]
pub fn det3(mat3x3: &[Vec<f64>]) -> f64 {
    *rob_det3(mat3x3).last().unwrap()
}


///Sign compute the 1.0 x sign of a number, return 0.0 when +/- 0.0
#[inline]
pub fn sign(n: f64) -> f64 {
    //handles +/- 0.0
    if n.abs() == 0.0 {
        return 0.0;
    }
    return n.signum();
}


/// Sign of a 2x2 determinant - robustly.
#[inline]
pub fn sign_of_det2(x1: f64, y1: f64, x2: f64, y2: f64) -> i32 {
    // returns -1 if the determinant is negative,
    // returns  1 if the determinant is positive,
    // returns  0 if the determinant is null.
    sign(det2(&[vec!(x1, y1), vec!(x2, y2)])) as i32
}


///Mid2D computes the mid coordinates
#[inline]
pub fn mid_2d(a: &[f64], b: &[f64]) -> (f64, f64) {
    (mid(a[X], b[X]), mid(a[Y], b[Y]))
}

///Mid computes the mean of two values
#[inline]
pub fn mid(x: f64, y: f64) -> f64 {
    (x + y) / 2.0
}


#[cfg(test)]
mod mutil_tests {
    use super::*;
    use std::f64::{INFINITY, NEG_INFINITY, MAX};

    const X: f64 = 0.1f64;
    const Y: f64 = 0.2f64;
    const Z: f64 = X + Y;
    const PRECISION: i32 = 8;


    #[test]
    fn test_indexes() {
        assert_eq!(super::X, 0usize);
        assert_eq!(super::Y, 1usize);
        assert_eq!(super::Z, 2usize);
    }


    #[test]
    fn test_feq() {
        assert!(0.3 != 0.1 + 0.2);
        assert!(feq(0.1 + 0.2, 0.3));
        assert!(feq(-0.000000087422776, -0.000000087422780));
        assert!(feq(-0.000000087422776, -0.000000087422780));
        assert!(feq(0.0000000000000001224646799147353207, 0.0000000000000001224646799147353177));
        assert!(feq(0.0, 0.0000000000000001224646799147353177));
        assert!(feq(-0.0000000000000001224646799147353207, 0.0));
        assert!(feq(-0.000000000000874227, 0.00000000000000012246));

        assert!(feq(INFINITY, INFINITY));
        assert!(feq(NEG_INFINITY, NEG_INFINITY));
        assert!(feq(MAX, MAX - 1e100));
        assert!(feq(0.1 + 0.2, 0.3));
        assert!(feq(0.3, 0.3));
        assert!(feq(0., -0.));
        assert!(feq(-0., 0.));
        assert!(feq(-0., -0.));
        assert!(!feq(0.3, 0.31));
        assert!(feq(X + Y, Z));
        assert!(feq(Z - 0.3, 0.0));
    }

    #[test]
    fn test_round() {
        assert_eq!(round_0(0.1 + 0.2), 0.);
        assert_eq!(round(0.1 + 0.2, 3), 0.3);
        assert_eq!(round(0.78591234, 3), 0.786);
        assert_eq!(round(0.78591234, 4), 0.7859);
        assert_eq!(round(3.1415926535897, PRECISION), round(PI, PRECISION));
        assert_eq!(round(0.523598775598299f64.to_degrees(), PRECISION), 30.0);
        assert_eq!(round(70.736380255432223f64.to_radians(), PRECISION),
                   round(1.2345827364, PRECISION));

        let sign = -0f64.signum();
        println!("the sign is {}", sign);
    }

    #[test]
    fn test_sign() {
        assert_eq!(sign(3.1415926535897), 1.0);
        assert_eq!(sign(-3.1415926535897), -1.0);
        assert_eq!(sign(0.0), 0.0);
        assert_eq!(sign(-0.0), 0.0);
    }

    #[test]
    fn test_det() {
        let mat3x3: Vec<Vec<f64>> = vec![
            vec!(0.617988, 0.27225, 0.398392),
            vec!(0.0552414, 0.258802, 0.800991),
            vec!(0.60112, 0.921029, 0.413371),
        ];

        let mat2x2: Vec<Vec<f64>> = vec![vec!(0.916756, 0.712766), vec!(0.546127, 0.498242)];
        assert_eq!(
            round(det2(&mat2x2), PRECISION),
            round(0.06750558567000006, PRECISION),
        );

        assert_eq!(
            round(det3(&mat3x3), PRECISION),
            round(-0.30663810342819653, PRECISION),
        );

        assert_eq!(sign_of_det2(0.916756, 0.712766, 0.546127, 0.498242), 1);
        assert_eq!(sign_of_det2(0., 0., 0., 0.), 0);
    }

    #[test]
    fn test_mid() {
        assert_eq!(mid_2d(&[0., 0.], &[100., 100.]), (50., 50.));
        assert_eq!(mid_2d(&[3., 6.], &[7., 9.]), (5.0, 7.5));
        assert_eq!(mid_2d(&[-3., -6.], &[7., 9.]), (2.0, 1.5));
    }
}
