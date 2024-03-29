pub use std::f64::consts::{E, SQRT_2, LN_2, PI, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4};
pub use robust_determinant::{det2 as rob_det2, det3 as rob_det3};
pub use bs_num::{num, Num, NumCast, Float, Feq, Flt, FloatConst, Numeric, Zero, One, max, min};

pub const TAU: f64 = 2.0f64 * PI;
pub const PRECISION: i32 = 12;
pub const EPSILON: f64 = 1.0e-12;

///Indexing X[0], Y[1], Z[2]
const X: usize = 0;
const Y: usize = 1;
const Z: usize = 2;

#[inline]
pub fn const_pi<T>() -> T where T: Flt {
    T::PI()
}

#[inline]
pub fn const_tau<T>() -> T where T: Flt {
    let k: T = num::cast(2).unwrap();
    k * const_pi::<T>()
}

#[inline]
pub fn const_epsilon<T>() -> T where T: Flt {
    let eps: T = num::cast(EPSILON).unwrap();
    eps
}

#[inline]
pub fn feq_eps<T>(a: T, b: T, eps: T) -> bool
    where T: Feq {
    a.feq_eps(b, eps)
}


#[inline]
pub fn feq<T>(a: T, b: T) -> bool
    where T: Feq {
    a.feq(b)
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
pub fn det2(mat2x2: &[[f64; 2]]) -> f64 {
    *rob_det2(mat2x2).last().unwrap()
}

#[inline]
pub fn det3(mat3x3: &[[f64; 3]]) -> f64 {
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
    sign(det2(&[[x1, y1], [x2, y2]])) as i32
}


///Computes the mid 2d coordinates
#[inline]
pub fn mid_2d(a: &[f64], b: &[f64]) -> (f64, f64) {
    (mid(a[X], b[X]), mid(a[Y], b[Y]))
}

///Computes the mid of 3d coordinates
#[inline]
pub fn mid_3d(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    (mid(a[X], b[X]), mid(a[Y], b[Y]), mid(a[Z], b[Z]))
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

    fn orientation_2d<T>(a: &[T], b: &[T], c: &[T]) -> T where T: Flt {
        let l = (a[1] - c[1]) * (b[0] - c[0]);
        let r = (a[0] - c[0]) * (b[1] - c[1]);
        let det = l - r;
        det
    }

    fn test_sample<T>(v: T) where T: Flt {
        let m = T::from(EPSILON).unwrap();
        let pi: T = T::PI();
        let other_pi: T = num::cast(PI).unwrap();
        println!("value {:?} = {:?}, {:?}, {:?}", v, pi, other_pi, m)
    }

    #[test]
    fn test_numerics() {
        assert_eq!(orientation_2d(&[0.1, 0.1], &[0.1, 0.1], &[0.3, 0.7]), 0.);
        test_sample(3.14f32);
        assert_eq!(const_epsilon::<f64>(), EPSILON);
        assert_eq!(const_pi::<f64>(), PI);
        assert_eq!(const_tau::<f64>(), TAU);
        assert_eq!(const_pi::<f32>(), std::f32::consts::PI);
        assert_eq!(const_tau::<f32>(), 2.0f32 * std::f32::consts::PI);
    }

    #[test]
    fn test_feq() {
        assert!(0.3 != 0.1 + 0.2);
        assert!(feq(0.1 + 0.2, 0.3));
        assert!(feq(1 + 2, 3));
        assert!((2i64 + 3i64).feq(5i64));
        assert!((0.1 + 0.2).feq(0.3));
        assert!((0.1f32 + 0.2f32).feq_eps(0.3f32, 1.0e-12f32));
        assert!(feq_eps(0.1 + 0.2, 0.3, 1.0e-12));
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
        let mat3x3: Vec<[f64; 3]> = vec![
            [0.617988, 0.27225, 0.398392],
            [0.0552414, 0.258802, 0.800991],
            [0.60112, 0.921029, 0.413371],
        ];

        let mat2x2 = vec![[0.916756, 0.712766], [0.546127, 0.498242]];
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

        assert_eq!(mid_3d(&[0., 0., 0.], &[100., 100., 100.]), (50., 50., 50.));
        assert_eq!(mid_3d(&[3., 6., -2.4], &[7., 9., 6.3]), (5.0, 7.5, 1.95));
        assert_eq!(mid_3d(&[-3., -6., 3.], &[7., 9., 9.]), (2.0, 1.5, 6.));
    }
}
